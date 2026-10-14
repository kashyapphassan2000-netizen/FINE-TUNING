# 91. PART K — GPU providers: what they offer and how to choose

:::honest What was checked (web, 21 September 2026)
Third-party comparisons from mid-2026 were read. They agree on the provider landscape and disagree on prices, which move monthly. This book lists the landscape and the **reported ranges** with their dates, and never uses a price in a calculation: `configs/providers.yaml` has empty price fields for the rate YOUR account is quoted.
:::

## 91.1 The landscape

|Type|Examples|Strength|Weakness|
|---|---|---|---|
|hyperscalers|AWS (g6 L4, p5 H100), GCP (g2 L4, a3 H100), Azure (ND H100 v5)|compliance, networking, your data is already there|highest per-GPU prices in the comparisons; quota requests|
|GPU clouds ("neoclouds")|CoreWeave, Lambda, Hyperstack, RunPod, Nebius|lower prices, NVLink/InfiniBand clusters, self-serve (most)|availability varies; some are contract-oriented|
|marketplaces|Vast.ai|lowest prices on commodity GPUs|reliability and SLA vary per host|
|serverless GPU|Modal, RunPod Serverless|pay per second, scale to zero|cold starts; less control|

## 91.2 Reported prices (snapshot, NOT for your budget)

Mid-2026 comparisons reported, for the same chip: H100 from roughly **$2/hour** (spot at specialised providers) to over **$10/hour** (hyperscaler on-demand); one tracker counted a **13.8× spread** for an H100 across 25 providers. Spot/interruptible capacity was reported 40-65% below on-demand, and 1-12 month commitments 20-40% below. Treat all of it as orientation only.

## 91.3 How to choose, in order

1. **Does the model fit?** VRAM per GPU and GPUs per node from `peftkit.plan` / `peftkit.capacity` (chapters 93-93).
2. **Interconnect.** Multi-GPU training or tensor-parallel serving needs NVLink inside a node; multi-node needs InfiniBand-class networking. PCIe-only cards are fine for single-GPU work and data-parallel replicas.
3. **Availability in your region**, today — the most common blocker in the comparisons.
4. **Billing granularity** (per second vs per hour) and **egress fees** (moving model weights and data out costs money on some clouds).
5. **Interruptible or not.** Training with checkpoints → spot is fine. A public API → on-demand or reserved.
6. **Compliance / data residency** (EU, healthcare, finance).
7. **Price** — your quote, entered into `configs/providers.yaml`.

@file configs/providers.yaml
