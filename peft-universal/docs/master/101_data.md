# 101. Data adaptation for any architecture

The data engineering of chapters 38-51 is universal: contracts, cleaning, dedup, PII, quality, leakage, splits. What changes per architecture is the **last mile**: processor → collator → labels → masks.

## 101.1 The last mile, per modality

|Modality / task|Raw record|Processor|Collator must|Labels|Mask|
|---|---|---|---|---|---|
|chat SFT (decoder)|`{"messages": [...]}`|tokenizer + chat template|pad right, build attention mask|token ids, prompt = -100|assistant-only|
|text classification (encoder)|`{"text", "label"}`|tokenizer, truncation|pad to longest in batch|class id|padding via attention mask|
|seq2seq|`{"source", "target"}`|tokenizer for both|pad source and target separately|target ids, pad = -100|decoder mask built by the model|
|image classification|`{"image_path", "label"}`|image processor (size, mean, std)|stack `pixel_values`|class id|—|
|detection / segmentation|image + boxes / masks|image processor with annotation conversion|variable boxes per image → lists|boxes+classes / mask|—|
|CTC ASR|`{"audio_path", "text"}`|feature extractor at the model's sampling rate + tokenizer|pad audio and labels separately|char/token ids, pad = -100|attention mask over audio|
|seq2seq ASR (Whisper)|`{"audio_path", "text"}`|log-mel feature extractor|stack features|target ids, pad = -100|—|
|vision-language|image + conversation|multimodal processor (inserts image tokens)|pad text, stack pixels, keep placeholder counts|answer tokens only|image + prompt = -100|
|preference (DPO)|`{"prompt", "chosen", "rejected"}`|tokenizer + template|pair padding|handled by the trainer|—|
|regression / tabular|`{"features", "target"}`|your normalisation (fit on TRAIN only)|stack tensors|float target|—|
|diffusion|`{"image" [,"caption"]}`|image transforms + text encoder|stack; noise added in the loss|the noise ε (created in the loss)|—|

## 101.2 A universal collator pattern

```py
def make_collator(pad_id: int, label_pad: int = -100):
    def collate(rows: list[dict]) -> dict:
        out = {}
        for key in rows[0]:
            vals = [r[key] for r in rows]
            if torch.is_tensor(vals[0]) and vals[0].dim() >= 1 and len({v.shape for v in vals}) > 1:
                pad = label_pad if key == "labels" else (0.0 if vals[0].is_floating_point() else pad_id)
                out[key] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=pad)
                if key in ("input_ids", "input_values"):
                    out["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                        [torch.ones(len(v), dtype=torch.long) for v in vals], batch_first=True)
            elif torch.is_tensor(vals[0]):
                out[key] = torch.stack(vals)
            else:
                out[key] = vals
        return out
    return collate
```

## 101.3 Verify the last mile, always

1. Decode one batch back to human form (text back from ids; image back from pixel_values with inverse normalisation; audio length vs label length).
2. Check the label/mask on that batch (chapter 22's `tokenize_inspect` for text; for CTC, frames vs label length).
3. Overfit that single batch (chapter 85.5). If it cannot, the last mile is wrong.
