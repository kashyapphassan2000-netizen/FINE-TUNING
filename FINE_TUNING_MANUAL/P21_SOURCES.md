# P21 — Sources

Prices and console steps were read on **14 October 2026**. Library facts were checked against the installed versions (torch 2.14.0, transformers 5.17.0, peft 0.21.0, trl 1.13.0, huggingface_hub 1.31.0, gradio 6.30.0) on the build machine.

## Documentation and price pages

| # | Source | Used for |
|---|---|---|
| 1 | Hugging Face Transformers docs: https://huggingface.co/docs/transformers | loading, chat templates, generation |
| 2 | PEFT docs: https://huggingface.co/docs/peft | LoRA/DoRA/AdaLoRA configs, merging |
| 3 | TRL docs: https://huggingface.co/docs/trl | SFT, DPO, GRPO, KTO, ORPO (experimental), prompt-completion masking |
| 4 | bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes | NF4, 8-bit/paged optimizers |
| 5 | Hugging Face Hub: storage and plans https://huggingface.co/pricing, Spaces docs https://huggingface.co/docs/hub/spaces | repos, Spaces hardware, secrets/variables |
| 6 | HyperStack GPU pricing: https://www.hyperstack.cloud/gpu-pricing | on-demand / reserved / spot prices, per-minute billing, free egress |
| 7 | RunPod pricing: https://www.runpod.io/pricing (page dated 8 Oct 2026) | Pod prices, storage prices |
| 8 | Vast.ai pricing: https://vast.ai/pricing | live marketplace from/median prices |
| 9 | RunPod SSH: https://docs.runpod.io/pods/configuration/use-ssh | basic (proxied, no SFTP) vs full SSH over exposed TCP |
| 10 | Vast.ai SSH/SCP: https://docs.vast.ai/guides/instances/connect/ssh | keys-only, direct vs proxy, `-P` for scp/sftp, tmux by default |
| 11 | HyperStack docs: https://infrahub-doc.nexgencloud.com/docs/intro | environments, keys, VMs |
| 12 | Gradio: https://www.gradio.app/docs (6.30.0) | `ChatInterface` |
| 13 | uv: https://docs.astral.sh/uv/ | environments |
| 14 | vLLM: https://docs.vllm.ai | serving, multi-LoRA, PagedAttention |
| 15 | lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness | benchmarks |
| 16 | DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/ | sharding |
| 17 | Accelerate: https://huggingface.co/docs/accelerate | launch configs, FSDP |
| 18 | PyTorch install matrix: https://pytorch.org/get-started/locally/ | torch builds per CUDA version |

## Papers (by chapter)

| Topic | Paper |
|---|---|
| scaling laws | Kaplan et al., *Scaling Laws for Neural Language Models*, 2020; Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), 2022; Besiroglu et al., *Chinchilla Scaling: A replication attempt*, 2024 |
| MTP | Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024; DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024 |
| objectives | Devlin et al., BERT, 2018; Raffel et al., T5, 2020; Tay et al., UL2, 2022 |
| data curation | Rae et al., Gopher (quality rules), 2021; Penedo et al., *The FineWeb Datasets*, 2024; Lee et al., *Deduplicating Training Data Makes Language Models Better*, 2022 |
| intrinsic dimension | Li et al., *Measuring the Intrinsic Dimension of Objective Landscapes*, 2018; Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020 |
| LoRA family | Hu et al., LoRA, 2021; Dettmers et al., QLoRA, 2023; Liu et al., DoRA, 2024; Zhang et al., AdaLoRA, 2023; Hayou et al., LoRA+, 2024; Kalajdzievski, rsLoRA, 2023 |
| instruction data | Wei et al., FLAN, 2021; Longpre et al., *The Flan Collection*, 2023; Wang et al., Self-Instruct, 2022; Taori et al., Alpaca, 2023; Teknium, OpenHermes 2.5 dataset card |
| CoT | Wei et al., *Chain-of-Thought Prompting*, 2022; DeepSeek-AI, DeepSeek-R1, 2025 |
| alignment | Ouyang et al., InstructGPT, 2022; Schulman et al., PPO, 2017; Gao et al., *Scaling Laws for Reward Model Overoptimization*, 2022; Rafailov et al., DPO, 2023; Shao et al., DeepSeekMath (GRPO), 2024; Yu et al., DAPO, 2025; Hong et al., ORPO, 2024; Ethayarajh et al., KTO, 2024 |
| attention / KV | Shazeer, MQA, 2019; Ainslie et al., GQA, 2023; DeepSeek-AI, DeepSeek-V2 (MLA), 2024; Dao et al., FlashAttention 1/2/3, 2022-2024; Kwon et al., PagedAttention (vLLM), 2023; Su et al., RoPE, 2021; Peng et al., YaRN, 2023 |
| synthetic tabular data | Patki et al., *The Synthetic Data Vault* (copulas), 2016; Nowok et al., *synthpop* (sequential/conditional synthesis), 2016; Shokri et al., membership inference, 2017 |
| forgetting | Kirkpatrick et al., *Overcoming catastrophic forgetting*, 2017; Luo et al., *An Empirical Study of Catastrophic Forgetting in LLMs during Continual Fine-tuning*, 2023 |
| model collapse | Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 2024 |
