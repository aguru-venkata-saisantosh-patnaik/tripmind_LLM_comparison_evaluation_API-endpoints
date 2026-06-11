# TripMind — Claude Code Context

## Project Summary
Autonomous multi-agent AI travel optimizer. Finds Price-Pivot Points (transit, accommodation, activity substitutions that save ≥5%) for Indian domestic travel.

## Current Status
| Phase | Status | Key Output |
|-------|--------|-----------|
| Phase 0 — Repo scaffold | ✅ Complete | Clean folder structure, shared utils |
| Phase 1 — Synthetic data | ✅ Complete | 5,000 validated records |
| Phase 2 — MCP servers + agents | ✅ Complete | 500 quality agent traces |
| Phase 3 — SLM training (3 models) | ✅ Complete | 3× 4.6GB GGUFs in models/ + HuggingFace backups |
| Phase 4 — Evals + red team | ✅ Complete | 92-sample eval + 45 red team runs, summary JSON |
| Phase 5 — FastAPI inference server | 🔲 In Progress | phase5_serving/ |

## Canonical Datasets
| File | Records | Notes |
|------|---------|-------|
| `data/synthetic/v2_20260608_085742.jsonl` | 5,000 | Phase 1 — gpt-4o-mini teacher |
| `data/traces/agent_traces_all.jsonl` | 500 | Phase 2 — DeepSeek teacher, quality-filtered |
| `data/training/ft_train.jsonl` | 4,749 | Phase 3 — fine-tuning train split |
| `data/training/distill_train.jsonl` | 449 | Phase 3 — distillation train split |

## Architecture
- **Phase 1**: Synthetic data generation (gpt-4o-mini → validated JSONL pairs)
- **Phase 2**: Multi-agent system (Supervisor + 3 workers) using MCP tool servers + DeepSeek function calling
- **Phase 3**: Three SLMs trained — fine-tuned (Colab T4, fp16), distilled (Lightning.ai A100, bf16), curriculum (A100, bf16). QLoRA r=8, Unsloth, GGUF Q4_K_M.
- **Phase 4**: Eval suite + red team (Groq Llama judge, sentence-transformers for intent alignment). 92 golden test cases × 4 models + 45 red team adversarial runs.
- **Phase 5**: FastAPI inference server (`phase5_serving/`) — REST endpoints for all 4 models, eval results API

## APIs Used
- OpenAI gpt-4o-mini — Phase 1 data generation (paid, complete)
- DeepSeek (`deepseek-chat`) — Phase 2 agent orchestration (free tier)
- Gemini 2.0 Flash (AI Studio) — Phase 4 eval judge (free, 1M tokens/day)
- OpenRouteService — routing/directions (free)
- Overpass API (OSM) — hotels, POIs, restaurants (free)
- duckduckgo-search — web search (free)
- Ollama — local SLM inference (free)

## Folder Structure
```
travel_project/
├── config.py                        # all shared constants
├── requirements.txt
├── utils/                           # logger, cache (used by all phases)
├── phase1_data_engine/              # data generation scripts
├── phase2_agents/                   # multi-agent orchestration
│   ├── agents/                      # analyst, concierge, optimizer
│   ├── mcp_servers/                 # routing, hotels, overpass, search servers
│   └── run.py                       # CLI entrypoint
├── phase3_training/                 # SLM training pipeline
│   ├── prepare_ft.py / prepare_distill.py / prepare_curriculum.py
│   ├── verify_datasets.py
│   └── notebooks/                   # 3 Colab notebooks + Ollama modelfiles
├── phase4_evals/                    # eval suite (complete)
├── phase5_serving/                  # FastAPI inference API (in progress)
├── data/
│   ├── synthetic/                   # Phase 1 output
│   ├── traces/                      # Phase 2 output (1 merged file)
│   ├── training/                    # Phase 3 input (6 JSONL files)
│   ├── evals/                       # Phase 4 output
│   └── seeds/                       # 50k seed personas
├── models/
│   ├── finetune/                    # tripmind-ft GGUF
│   ├── distill/                     # tripmind-distill GGUF
│   └── curriculum/                  # tripmind-curriculum GGUF
└── logs/phase1/ … phase4/
```

## Development Rules
- Never hardcode API keys — always load from `.env` via `python-dotenv`
- Never hardcode budget tiers — always import from `config.BUDGET_TIERS`
- Always use `utils/logger.py` for logging, never `print()` in production scripts
- Always use `utils/cache.py` for external API calls
- Phases execute ONE AT A TIME — get explicit user approval before starting the next phase

## Compute Constraints
- MacBook Air 8GB RAM — local inference via Ollama only, no GPU training
- Phase 3 ft: Colab T4 (fp16, seq_len=512) | distill + curriculum: Lightning.ai A100 (bf16, seq_len=16384)
- Phase 2 agents run locally (no GPU needed)

## Phase 2 — Starting MCP Servers + Agents
```bash
# Terminal 1-4: start MCP servers
python phase2_agents/mcp_servers/routing_server.py
python phase2_agents/mcp_servers/hotels_server.py
python phase2_agents/mcp_servers/overpass_server.py
python phase2_agents/mcp_servers/search_server.py

# Terminal 5: run agents
python phase2_agents/run.py --limit 25 --concurrency 3
```

## Phase 3 — Register Models with Ollama (run from project root)
```bash
# GGUFs already in models/ — just register
ollama create tripmind-ft         -f phase3_training/notebooks/modelfiles/Modelfile.ft
ollama create tripmind-distill    -f phase3_training/notebooks/modelfiles/Modelfile.distill
ollama create tripmind-curriculum -f phase3_training/notebooks/modelfiles/Modelfile.curriculum

# HuggingFace backups: agurusantosh/tripmind-{ft,distill,curriculum}-{lora,gguf}
```
