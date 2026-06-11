# Pushing Model Cards to HuggingFace Hub

The three `.md` files in this folder are ready-to-use model cards. Copy each one to the
corresponding HuggingFace repo as `README.md`.

## Method 1 — huggingface_hub CLI (recommended)

```bash
pip install huggingface_hub
huggingface-cli login   # enter your HF token (write access)

# tripmind-ft
huggingface-cli upload agurusantosh/tripmind-ft-gguf \
  phase3_training/hf_cards/tripmind_ft.md README.md

# tripmind-distill
huggingface-cli upload agurusantosh/tripmind-distill-gguf \
  phase3_training/hf_cards/tripmind_distill.md README.md

# tripmind-curriculum
huggingface-cli upload agurusantosh/tripmind-curriculum-gguf \
  phase3_training/hf_cards/tripmind_curriculum.md README.md
```

## Method 2 — HuggingFace web UI

1. Go to each model repo on huggingface.co
2. Click **Files** → **README.md** → **Edit**
3. Paste the contents of the corresponding `.md` file
4. Click **Commit changes**
