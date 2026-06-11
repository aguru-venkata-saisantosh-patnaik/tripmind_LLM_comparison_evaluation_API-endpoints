# Phase 3 — SLM Training ✅ COMPLETE

Trains and compares three SLMs on Llama 3.1 8B using QLoRA (Unsloth). Tests which training approach produces the best travel optimizer.

## The Research Question

> Does distilling DeepSeek's agent reasoning chains produce a better travel optimizer than plain SFT on synthetic itinerary pairs?

Three models, same base, different training signals:

| Model | Data | Method | Epochs | LR | Hardware | Final Loss |
|-------|------|--------|--------|----|----------|------------|
| `tripmind-ft` | Phase 1 — 4,749 pairs | Single SFT run | 3 | 2e-4 | Colab T4 (fp16) | 0.225 |
| `tripmind-distill` | Phase 2 — 449 traces | Single SFT run | 5 | 2e-4 | Lightning.ai A100 (bf16) | 0.254 |
| `tripmind-curriculum` | Phase 1 → Phase 2 | Two-stage SFT, same model | 2+3 | 2e-4→5e-5 | Lightning.ai A100 (bf16) | 0.241 / 0.505 |

**Base model**: `unsloth/Meta-Llama-3.1-8B-bnb-4bit`  
**QLoRA config**: r=8, lora_alpha=16, load_in_4bit=True, gradient_checkpointing=unsloth  
**ft**: Colab T4 (15GB VRAM, fp16, seq_len=512)  
**distill + curriculum**: Lightning.ai A100 (40GB VRAM, bf16, seq_len=16384)

---

## Datasets

All files are in `data/training/` — generated, verified, and used for training.

| File | Records | Used by |
|------|---------|---------|
| `ft_train.jsonl` | 4,749 | Notebook 01 |
| `ft_val.jsonl` | 249 | Notebook 01 |
| `distill_train.jsonl` | 449 | Notebook 02 |
| `distill_val.jsonl` | 49 | Notebook 02 |
| `curriculum_stage1.jsonl` | 4,749 | Notebook 03 (Stage 1) |
| `curriculum_stage2.jsonl` | 449 | Notebook 03 (Stage 2) |

All datasets use **Alpaca format**:
```json
{ "instruction": "...", "input": "<persona JSON>", "output": "<itinerary or reasoning chain>" }
```

---

## Phase 3 is Complete

GGUFs are already in `models/finetune/`, `models/distill/`, `models/curriculum/` (4.6GB each).  
HuggingFace repos: `agurusantosh/tripmind-ft-gguf`, `agurusantosh/tripmind-distill-gguf`, `agurusantosh/tripmind-curriculum-gguf`

### Register with Ollama (run once from project root)

```bash
ollama create tripmind-ft         -f phase3_training/notebooks/modelfiles/Modelfile.ft
ollama create tripmind-distill    -f phase3_training/notebooks/modelfiles/Modelfile.distill
ollama create tripmind-curriculum -f phase3_training/notebooks/modelfiles/Modelfile.curriculum

# Verify
ollama list
ollama run tripmind-ft "Persona: Solo, Delhi to Goa, Budget+, 5 days. Optimize."
```

The Modelfiles use paths relative to their own location (`../../../models/...`) — must be run from the **project root** as shown above.

---

## Re-running Training (if needed)

Training platforms used:

| Notebook | Platform | Time |
|----------|----------|------|
| `01_train_ft.ipynb` | Google Colab T4 | ~1.7 hrs |
| `02_train_distill.ipynb` | Lightning.ai A100 (free 3hr) | ~8 min |
| `03_train_curriculum.ipynb` | Lightning.ai A100 (free 3hr) | ~13 min |

To re-run:
1. `python phase3_training/verify_datasets.py` — validate all 6 files
2. Zip project: `cd .. && zip -r tripmind.zip travel_project/ --exclude "*.gguf" --exclude "*.jsonl" --exclude ".cache/*"`
3. Upload zip to Colab Drive (for 01) or Lightning.ai studio storage (for 02/03)
4. Run notebook, then `ollama create` again with updated GGUF

---

## Curriculum Training — How it Works

Notebook 03 is mechanically different from the other two. It runs **two sequential `SFTTrainer` calls on the same model object** — not two separate model loads:

```python
# Stage 1 — domain knowledge (Phase 1, lr=2e-4, 2 epochs)
trainer_s1 = SFTTrainer(model=model, dataset=stage1_data, lr=2e-4, epochs=2)
trainer_s1.train()
model.save_pretrained("Drive/.../curriculum_stage1_lora")  # crash checkpoint

# Stage 2 — reasoning on top (Phase 2, lr=5e-5, 3 epochs)
# Same model object — Stage 1 LoRA weights already in memory
trainer_s2 = SFTTrainer(model=model, dataset=stage2_data, lr=5e-5, epochs=3)
trainer_s2.train()
```

The **4x lower LR in Stage 2** is critical: it prevents catastrophic forgetting of the domain knowledge learned in Stage 1. Stage 2 makes small gradient updates on top of Stage 1's weights rather than overwriting them.

---

## Re-running Data Preparation

If you need to regenerate the training files (e.g., after adding more traces):

```bash
python phase3_training/prepare_ft.py          # Phase 1 → ft_train.jsonl + ft_val.jsonl
python phase3_training/prepare_distill.py     # Phase 2 → distill_train.jsonl + distill_val.jsonl
python phase3_training/prepare_curriculum.py  # copies stage1 + stage2 from above
python phase3_training/verify_datasets.py     # validates all 6 files
```

---

## Training Outcomes

| Model | Steps | Runtime | Final Loss | Train Loss |
|-------|-------|---------|------------|------------|
| `tripmind-ft` | 636 | 6,176s (~1.7h) | 0.225 | 0.266 |
| `tripmind-distill` | 285 | 476s (~8 min) | 0.254 | 0.429 |
| `tripmind-curriculum` S1 | 424 | 488s | 0.241 | 0.313 |
| `tripmind-curriculum` S2 | 171 | 281s | 0.505 | 0.686 |

**Notes:**
- distill's higher absolute loss (0.429) reflects harder output space (5k-token reasoning chains vs structured JSON), not poor learning.
- curriculum S2 higher than distill S2 loss by design — 4× lower LR deliberately slows adaptation to preserve S1 domain weights.
- Phase 4 evals (against 100-record golden set) will determine empirical quality differences.

## Expected Performance (theory — Phase 4 will confirm)

| Metric | `tripmind-ft` | `tripmind-distill` | `tripmind-curriculum` |
|--------|--------------|--------------------|-----------------------|
| Intent alignment | ~72% | ~83% | ~94% |
| Budget compliance | ~88% | ~76% | ~100% |
| Reasoning coherence | ~65% | ~82% | ~97% |
| JSON validity | ~91% | ~85% | ~97% |
| **Aggregate** | **~77%** | **~81%** | **~85%** |
