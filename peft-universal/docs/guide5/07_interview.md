# 8. Fifteen interview questions, with crisp answers

1. **What is SFT?** Supervised training on prompt → ideal-answer pairs, with loss on the answer only, to make a pretrained model follow instructions in a chosen format.
2. **Where does it sit in post-training?** After pretraining (and any continued pretraining), before preference optimization. Every later stage assumes it.
3. **Why mask the prompt?** The model never generates the prompt; training on it wastes capacity and teaches it to continue user turns.
4. **Instruction tuning vs SFT?** Instruction tuning is SFT across many tasks so the model generalises to "follow instructions", rather than learning one task.
5. **What did FLAN show?** Held-out performance grows log-linearly with the number of fine-tuning tasks; FLAN 2022 has 1,836 tasks, and templates per task matter as much as tasks.
6. **What did Alpaca show?** That distilling a stronger model gives usable instruction data cheaply (52K records from 175 seeds) — and that licences (CC BY-NC 4.0) travel with the data.
7. **What is the risk of compiled chat collections like OpenHermes-style mixes?** Mixed quality and inherited licences: the compilation is only as permissive as its most restrictive source.
8. **How is a multi-turn conversation trained?** As one example, with every assistant turn a target and all user/system turns masked, via `{% generation %}` markers.
9. **How do you stop a model repeating the system prompt?** Mask it (assistant-only loss) and vary it across examples.
10. **What is CoT fine-tuning?** Putting reasoning steps in the assistant target so the model produces them and conditions its answer on them.
11. **Biggest practical risk with CoT data?** Unverified traces: correct answers reached by wrong reasoning teach the model to fake reasoning. Filter by final-answer correctness.
12. **Cost of CoT?** Targets are 3-5× longer: more training tokens, more output tokens, higher latency per request.
13. **How much domain data do you need?** Unknown until measured: plot the eval curve at 25/50/100% of your data. Coverage of intents, phrasings and entities matters more than raw volume.
14. **How do you detect catastrophic forgetting after domain SFT?** Run a general suite before and after; a domain gain with a large general drop is usually a bad trade (Volume 2, chapter 10).
15. **Your SFT model is confidently wrong about a policy detail. Fix?** Do not fine-tune the fact in: retrieve it. Train the behaviour of using context and of saying "not in the documents".
