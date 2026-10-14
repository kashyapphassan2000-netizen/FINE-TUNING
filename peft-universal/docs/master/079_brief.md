# 79. Appendix: the brief this book follows

Every rule of the instruction brief you gave, and where this edition honours it.

|Brief rule|Where|
|---|---|
|Classify the topic (PROJECT / CONCEPT / TOOL)|chapter 0: PROJECT with full CONCEPT chapters|
|Check latest versions and record the date|chapter 0.3|
|Read package source for fast-moving APIs|chapters 19-29 and 41-43 cite source-verified facts (e.g. `warmup_ratio` removed in transformers 5, AdaLoRA callback, vLLM DoRA refusal)|
|Find dependency conflicts, design separate environments|chapter 31 and runbook step 23 (`.venv-train` vs `.venv-serve`)|
|Say plainly what is impossible or partial|chapters 1.3, 3 (limitations), 22 (trillion scale), 53-54 (local server limits)|
|Write every file; py_compile, bash -n, pytest; start the API end to end|the ZIP; chapter 77; `scripts/rehearsal_cpu.py` starts the server and gateway and calls them|
|Mocks for GPU/paid parts|fake OpenAI server, fake vLLM, moto S3, tiny offline model (chapter 77)|
|List what was executed vs checked|chapter 0.3|
|Cover page, clickable contents, page numbers, bookmarks|this PDF|
|Legend of machine labels|chapter 0.1|
|Requirement → implementation map|chapter 18|
|Architecture with honest limitations|chapter 17|
|Accounts and keys as browser steps|chapter 30|
|Laptop setup from zero, .env, git|chapter 31, runbook stage 2|
|One section per pipeline stage in build order|Parts C-H|
|FakeCloud first, then AWS, budget and cleanup|chapter 72|
|Deliverables, README, resume bullets with placeholders|chapter 76|
|Final verification loop + troubleshooting|chapters 77-78|
|Every stage: WHY box, FILE, commands, output, WARNING, CHECKPOINT|throughout|
|No invented numbers, prices or benchmarks|all numbers are from runs or verified sources; no prices anywhere|
|ZIP byte-identical to the PDF files|the PDF embeds files from the repository at build time|
