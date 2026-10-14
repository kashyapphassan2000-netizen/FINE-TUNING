# P16 — Domain playbooks: what changes for healthcare, legal, finance, support, code, manufacturing, education

The pipeline (P02-P14) is identical for every domain. What changes is **the data, the eval suites, the safety suite, the privacy rules and the definition of "correct"**. Use the row for your domain to fill P05's plan.

> **Not legal or medical advice.** The regulation column names what to check with your compliance owner; it does not tell you that you comply.

## The universal checklist (every domain)

- [ ] behaviour vs facts split written down (P05 §1): facts that change go to retrieval
- [ ] eval set built first, with held-out entities, from real users or experts (P05 §2)
- [ ] a **safety suite** specific to the domain's harms (rows below)
- [ ] PII policy decided: scrub, drop or keep with consent; identifiers removed before synthesis (P07 §2)
- [ ] licences of every data source recorded (`peftkit.data.lineage`)
- [ ] 5-30% replay data (general ability) mixed in
- [ ] a human expert reads 30 training rows and 30 model answers before release

---

## Healthcare / clinical operations

| Item | Guidance |
|---|---|
| good fine-tuning targets | documentation formatting (notes → structured summaries), coding assistance (ICD/CPT suggestions **for human review**), patient-message drafting, scheduling and operations (the P07 no-show example), extraction from referrals and lab reports |
| bad targets | autonomous diagnosis or treatment decisions; "knowing" drug doses (retrieve from a maintained formulary, cite it) |
| data | de-identified notes (remove the identifiers your regulation lists: names, dates more specific than the year, contact details, record numbers…), synthetic records measured against real ones (P07), public sources with their terms: OpenFDA (allowed), PhysioNet/MIMIC (credentialed, strict DUA, **no redistribution**) |
| eval | exact fields (codes, dates, doses) with `must_match`; clinician-written questions; per-specialty segments; refusal when information is missing |
| safety suite | requests for diagnosis/dosage outside scope, emergency symptoms (must route to emergency care), patient-identifying output, overconfident answers |
| benchmarks | MedQA / PubMedQA / MedMCQA-style tasks in lm-eval (check task names for your version), plus a general suite |
| regulation to check | HIPAA (US), GDPR special-category data (EU), medical-device rules if it influences clinical decisions (EU MDR / FDA SaMD), local health-data laws |
| base model tip | a strong general instruct model + RAG over guidelines beats a small "medical" model trained on stale text |

## Legal

| Item | Guidance |
|---|---|
| good targets | clause extraction and classification, contract summary in a house format, issue spotting checklists, drafting first versions from templates, citation formatting |
| bad targets | memorising case law or statutes (they change; retrieve them, cite them); final legal opinions |
| data | your firm's own documents (licence `owned`), public court decisions where reuse is permitted, templates; strip client identifiers |
| eval | clause-level precision/recall, exact citation format (`must_match`), "not in the document" refusals, jurisdiction segments |
| safety suite | requests for advice outside the jurisdiction, confidential-information leakage, fabricated citations (check every citation exists) |
| regulation to check | client confidentiality / privilege, data-protection law, professional-conduct rules on AI use |

## Finance / insurance / banking

| Item | Guidance |
|---|---|
| good targets | document extraction (invoices, claims, KYC forms) to JSON, transaction categorisation, customer-message drafting in a compliant tone, report summarisation |
| bad targets | investment recommendations; numbers the model "remembers" (prices, rates: retrieve them) |
| data | transaction/claims exports with identifiers removed → synthetic data with the rare events (fraud, defaults) deliberately generated and the base rate recorded (P07 §7) |
| eval | field-level exact match, numeric tolerance checks, rare-class recall, per-product segments, **AUC not accuracy** on imbalanced labels (P07 §6) |
| safety suite | requests for personalised investment advice, discriminatory decisions (counterfactual tests: change one protected attribute, `perturb.counterfactual`), leaking account data |
| regulation to check | consumer-protection and fair-lending rules, model-risk-management guidance, PCI DSS for card data, GDPR, the EU AI Act's high-risk categories (credit scoring) |

## Customer support

| Item | Guidance |
|---|---|
| good targets | the house tone, troubleshooting flows, ticket classification and routing, reply drafting from retrieved KB articles |
| data | resolved tickets (subject + body → resolution) via `peftkit.data.records --task template`, KB documents via `peftkit.data.build`, multi-turn conversations |
| eval | resolution correctness on held-out products, escalation when needed, tone rubric with a calibrated judge |
| safety suite | refunds/promises the policy does not allow, prompt-injection inside customer messages, leaking other customers' data |

## Software / code

| Item | Guidance |
|---|---|
| good targets | an internal codebase's conventions and APIs (CPT on the repo + SFT on tasks), code review comments, test generation, a proprietary DSL |
| data | your repos (licence `owned`; check third-party code licences), task pairs built from commits (message → diff), docs |
| eval | **execution**: unit tests pass/fail (also the perfect GRPO verifier, P13), plus HumanEval/MBPP-style benchmarks for forgetting |
| safety suite | secrets in output (scan for keys), insecure patterns, licence-incompatible snippets |

## Manufacturing / industrial / IoT

| Item | Guidance |
|---|---|
| good targets | troubleshooting from manuals (the P06 pump example), maintenance-log summarisation, fault classification from sensor summaries, work-instruction drafting |
| data | manuals, service logs, sensor data (time series → features → synthetic with seasonality and shocks preserved: `synthkit.profile` with `time_col`/`value_col`, lab 43) |
| eval | correct procedure steps in order, units and limits exact (`must_include`), rare-failure recall |
| safety suite | unsafe procedures (lock-out/tag-out skipped), operating outside limits |

## Education

| Item | Guidance |
|---|---|
| good targets | tutoring in a pedagogy style (hints before answers), grading rubrics, question generation per curriculum level |
| eval | correctness of worked solutions (verifiable for maths: GRPO-friendly), level-appropriateness, refusal to just hand over exam answers when the policy says so |
| safety suite | age-inappropriate content, student data exposure |
| regulation to check | student-data privacy laws (e.g. FERPA in the US, GDPR for minors) |

## Multilingual / a new language

| Item | Guidance |
|---|---|
| when CPT is justified | a low-resource language the base model writes poorly (T01 §2.4): CPT on 100M+ tokens of clean text, then SFT |
| tokenizer check | count tokens per word: if your language costs 3-5× more tokens than English, context and cost scale with it (P03 Section 2). Prefer a base model whose tokenizer covers the script |
| eval | native-speaker test set; per-dialect segments; a general suite in **both** languages (forgetting) |
