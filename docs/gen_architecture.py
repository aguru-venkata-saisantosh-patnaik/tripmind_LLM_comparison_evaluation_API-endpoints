"""TripMind architecture diagram — dark theme matching the Phase 4 eval charts.
Renders a layered systems diagram: data generation, the agent->MCP->API call
stack, QLoRA training, evaluation, and FastAPI serving. Pure matplotlib."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Arc, RegularPolygon
from pathlib import Path
import numpy as np

# ── Palette (identical to results_analysis.ipynb) ────────────────────────────
BG    = '#1a1a2e'
CARD  = '#16213e'
INNER = '#1f2a48'
INNER2= '#24345c'
TEXT  = '#e6e8ef'
DIM   = '#98a1ba'

C1 = '#fb923c'   # orange — data
C2 = '#4ade80'   # green  — agents
C3 = '#c084fc'   # purple — training
C4 = '#f87171'   # red    — eval
C5 = '#38bdf8'   # blue   — serving
GREY = '#64748b'

# Exact model colours from the eval charts
M_FT, M_DIST, M_CURR, M_BASE = '#4fc3f7', '#81c784', '#ffb74d', '#ef5350'

ARROW = '#cdd3e0'
THIN  = '#7c8aa5'
PILL  = '#243054'
PILLE = '#46568a'

# ── Canvas ───────────────────────────────────────────────────────────────────
FW, FH = 15.0, 25.5
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.set_aspect('equal'); ax.axis('off')

# ── Helpers ──────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.5, r=0.16, z=2, ls='solid'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                 facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))

def txt(x, y, s, size=10, color=TEXT, weight='normal', ha='center', va='center', z=6, style='normal'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=z, fontfamily='DejaVu Sans', fontstyle=style)

def phase(x, y, w, h, color, num, title, sub=''):
    hb = 0.64
    rbox(x, y, w, h, CARD, color, lw=2.0, r=0.26, z=2)
    rbox(x, y + h - hb, w, hb, color, color, lw=0, r=0.26, z=3)
    ax.add_patch(Rectangle((x, y + h - hb), w, hb - 0.26, facecolor=color, edgecolor='none', zorder=3))
    ax.plot([x, x + w], [y + h - hb, y + h - hb], color=BG, lw=1.2, zorder=4)
    cy = y + h - hb / 2
    ax.add_patch(plt.Circle((x + 0.52, cy), 0.235, color='white', zorder=5))
    txt(x + 0.52, cy, str(num), size=14, color=color, weight='bold', z=6)
    txt(x + 0.95, cy, title, size=12, color='#0f1626', weight='bold', ha='left', z=6)
    if sub:
        txt(x + w - 0.3, cy, sub, size=9, color='#0f1626', weight='bold', ha='right', z=6)

def node(x, y, w, h, ec, title, lines=(), fc=INNER, tcolor=None, lw=1.5, z=4, tsize=10.5):
    rbox(x, y, w, h, fc, ec, lw=lw, r=0.15, z=z)
    txt(x + w/2, y + h - 0.33, title, size=tsize, color=tcolor or ec, weight='bold', z=z+1)
    for i, ln in enumerate(lines):
        txt(x + w/2, y + h - 0.66 - i*0.32, ln, size=8.6, color=DIM, z=z+1)

def cylinder(cx, by, w, h, ec, label, sub='', body=INNER, top=INNER2):
    eh = 0.34
    bcy, tcy = by + eh/2, by + h - eh/2
    ax.add_patch(Rectangle((cx-w/2, bcy), w, h-eh, facecolor=body, edgecolor='none', zorder=4))
    ax.plot([cx-w/2, cx-w/2], [bcy, tcy], color=ec, lw=1.6, zorder=5)
    ax.plot([cx+w/2, cx+w/2], [bcy, tcy], color=ec, lw=1.6, zorder=5)
    ax.add_patch(Arc((cx, bcy), w, eh, theta1=180, theta2=360, edgecolor=ec, lw=1.6, zorder=5))
    ax.add_patch(Ellipse((cx, tcy), w, eh, facecolor=top, edgecolor=ec, lw=1.6, zorder=6))
    txt(cx, by + h/2 + 0.04, label, size=9.6, color=ec, weight='bold', z=7)
    if sub:
        txt(cx, by + h/2 - 0.30, sub, size=8.0, color=DIM, z=7)

def hexagon(cx, cy, r, ec, label, sub=''):
    ax.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=r, orientation=np.pi/6,
                 facecolor=INNER, edgecolor=ec, lw=1.6, zorder=4))
    txt(cx, cy + 0.07, label, size=8.8, color=ec, weight='bold', z=6)
    if sub:
        txt(cx, cy - 0.20, sub, size=7.4, color=DIM, z=6)

def big_arrow(xc, y_top, y_bot, label):
    ax.annotate('', xy=(xc, y_bot), xytext=(xc, y_top),
                arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=3.8, mutation_scale=28))
    pw = 0.158*len(label) + 0.7; ph = 0.48
    mx = xc + 0.28 + pw/2; my = (y_top + y_bot)/2
    rbox(mx-pw/2, my-ph/2, pw, ph, PILL, PILLE, lw=1.2, r=0.18, z=7)
    txt(mx, my, label, size=9.4, color=TEXT, weight='bold', z=8)

def arr_h(x1, x2, y, color=THIN, lw=1.7):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=15))

def loop(xc, y1, y2, label):
    """Double-headed vertical arrow (request/response) with a side label."""
    ax.annotate('', xy=(xc, y2), xytext=(xc, y1),
                arrowprops=dict(arrowstyle='<|-|>', color=C2, lw=2.2, mutation_scale=16))
    pw = 0.148*len(label) + 0.5
    rbox(xc + 0.35, (y1+y2)/2 - 0.21, pw, 0.42, '#1c3326', '#2f6b44', lw=1.1, r=0.16, z=7)
    txt(xc + 0.35 + pw/2, (y1+y2)/2, label, size=8.4, color='#bbf7d0', weight='bold', z=8)

def chip(cx, cy, w, text, color):
    rbox(cx-w/2, cy-0.17, w, 0.34, '#0f1a30', color, lw=1.1, r=0.15, z=5)
    txt(cx, cy, text, size=7.8, color=color, weight='bold', z=6)

# ── Title ────────────────────────────────────────────────────────────────────
txt(FW/2, 24.95, 'TripMind — System Architecture', size=18.5, color=TEXT, weight='bold')
txt(FW/2, 24.5, 'multi-agent travel optimizer  ·  data → agents → training → evaluation → serving',
    size=10, color=DIM, style='italic')
ax.plot([FW/2-2.4, FW/2+2.4], [24.18, 24.18], color=C5, lw=2.4, zorder=4)

# ════════════════════════ PHASE 1 — DATA GENERATION ══════════════════════════
Y, H = 22.0, 1.9
phase(0.3, Y, 14.4, H, C1, 1, 'Synthetic Data Engine', 'OpenAI gpt-4o-mini · $4')
node(0.6, Y+0.18, 3.5, 1.0, C1, 'GPT-4o-mini teacher',
     ('5,000 prompts · 20 cities', '5 budgets · 8 intents'))
arr_h(4.1, 4.7, Y+0.68)
node(4.7, Y+0.18, 4.0, 1.0, C1, '3-Gate Validator',
     ('hostel · savings ≥ 5% · budget', '~12% rejected · checkpoint-safe'))
arr_h(8.7, 9.35, Y+0.68)
cylinder(11.55, Y+0.16, 4.6, 1.06, C1, '5,000 training pairs',
         '(baseline, optimized) + pivot · Alpaca')

big_arrow(FW/2, Y, Y - 0.95, 'persona → itinerary pairs')

# ════════════════════════ PHASE 2 — AGENTIC PIPELINE ═════════════════════════
Y, H = 12.8, 8.05
phase(0.3, Y, 14.4, H, C2, 2, 'Multi-Agent Orchestration', 'DeepSeek V4 Flash · $4')

# Supervisor
sup_y = Y + H - 1.55
rbox(3.9, sup_y, 7.2, 0.72, INNER2, C2, lw=1.8, r=0.16, z=4)
txt(5.0, sup_y+0.36, 'Supervisor', size=11, color=C2, weight='bold', z=5)
txt(8.6, sup_y+0.46, 'async · concurrency = 3 · checkpoint-resume', size=8.3, color=DIM, z=5)
txt(8.6, sup_y+0.18, 'quality filter → 500 clean traces (from 545)', size=8.3, color=DIM, z=5)

# down from supervisor into agent chain
ax.annotate('', xy=(FW/2, sup_y-0.32), xytext=(FW/2, sup_y),
            arrowprops=dict(arrowstyle='-|>', color=THIN, lw=1.7, mutation_scale=15))

# Agent chain (3 boxes, left→right flow)
ag_y, ag_h, aw = sup_y - 1.85, 1.4, 4.3
agents = [('Analyst', 'identifies cost drivers', '→ cost_report', C2),
          ('Concierge', 'finds cheaper swaps', '→ substitutions', C2),
          ('Optimizer', 'builds final itinerary', '→ pivot_analysis', C2)]
axs = [0.7, 5.3, 9.9]
for (name, role, out, col), bx in zip(agents, axs):
    rbox(bx, ag_y, aw, ag_h, INNER, col, lw=1.6, r=0.16, z=4)
    txt(bx+aw/2, ag_y+ag_h-0.34, name, size=11.5, color=col, weight='bold', z=5)
    txt(bx+aw/2, ag_y+ag_h-0.70, role, size=8.7, color=TEXT, z=5)
    txt(bx+aw/2, ag_y+0.30, out, size=9.4, color=col, weight='bold', z=5)
arr_h(axs[0]+aw, axs[1], ag_y+ag_h/2, color=C2, lw=2.0)
arr_h(axs[1]+aw, axs[2], ag_y+ag_h/2, color=C2, lw=2.0)

# request/response loop: agents <-> MCP
loop(FW/2, ag_y, ag_y-0.95, 'function call  ↕  tool result')

# MCP layer label + 4 servers
txt(FW/2, ag_y-1.18, 'MCP TOOL LAYER  ·  official mcp library  ·  SSE transport  ·  @api_cache (ttl 86400s)',
    size=8.6, color=C2, weight='bold', z=5)
mcp_y = ag_y - 2.45
mcp = [('routing :8001', 'get_route'), ('hotels :8002', 'search_hotels'),
       ('overpass :8003', 'search_pois'), ('search :8004', 'web_search')]
mw = 3.25
mxs = [0.7, 4.3, 7.9, 11.5]
for (n, t), bx in zip(mcp, mxs):
    rbox(bx, mcp_y, mw, 0.92, INNER, C2, lw=1.5, r=0.14, z=4)
    txt(bx+mw/2, mcp_y+0.60, n, size=9.3, color=C2, weight='bold', z=5)
    txt(bx+mw/2, mcp_y+0.28, t, size=8.2, color=DIM, z=5)

# loop: MCP <-> external APIs
loop(FW/2, mcp_y, mcp_y-0.85, 'HTTP  ↕  JSON  (cached)')

# External API layer label + hexagons
txt(FW/2, mcp_y-1.05, 'EXTERNAL DATA SOURCES  ·  free APIs', size=8.6, color=DIM, weight='bold', z=5)
hex_y = mcp_y - 1.95
apis = [('ORS', 'routing'), ('Overpass', 'hotels'), ('Overpass', 'POIs'), ('DuckDuckGo', 'search')]
for (n, s), bx in zip(apis, mxs):
    hexagon(bx+mw/2, hex_y, 0.62, GREY, n, s)

big_arrow(FW/2, Y, Y - 0.95, '500 reasoning traces')

# ════════════════════════ PHASE 3 — TRAINING ═════════════════════════════════
Y, H = 8.35, 3.45
phase(0.3, Y, 14.4, H, C3, 3, 'QLoRA Fine-Tuning', 'Llama 3.1 8B · Unsloth · r=8')

# data lake (two cylinders) on the left
txt(2.0, Y+H-1.0, 'TRAINING DATA', size=8.2, color=DIM, weight='bold', z=5)
cylinder(2.0, Y+1.35, 2.3, 0.95, C1, '5,000 pairs', 'Phase 1')
cylinder(2.0, Y+0.28, 2.3, 0.95, C2, '500 traces', 'Phase 2')

# arrow from data lake into the model cards
ax.annotate('', xy=(3.85, Y+1.5), xytext=(3.25, Y+1.5),
            arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=2.4, mutation_scale=18))

# three model cards
cards = [
    (M_FT,   'tripmind-ft',         'trained on 5,000 pairs',          ('SFT · Colab T4 · fp16', '3 epochs · loss 0.225')),
    (M_DIST, 'tripmind-distill',    'trained on 500 traces',           ('KD · A100 · bf16', '5 epochs · loss 0.254')),
    (M_CURR, 'tripmind-curriculum', '5,000 pairs → 500 traces',        ('2-stage · A100 · lr decay 4×', 'loss 0.241 / 0.505')),
]
cw = 3.1
cxs = [3.9, 7.3, 10.7]
for (col, name, data, lines), bx in zip(cards, cxs):
    rbox(bx, Y+0.28, cw, H-1.05, INNER, col, lw=1.7, r=0.16, z=4)
    txt(bx+cw/2, Y+H-1.05, name, size=10.5, color=col, weight='bold', z=5)
    chip(bx+cw/2, Y+H-1.5, cw-0.5, data, col)
    for i, ln in enumerate(lines):
        txt(bx+cw/2, Y+0.92-i*0.30, ln, size=8.3, color=DIM, z=5)
    txt(bx+cw/2, Y+0.34, '4.6 GB GGUF Q4_K_M', size=8.0, color=DIM, weight='bold', z=5)

big_arrow(FW/2, Y, Y - 0.95, '3 fine-tuned models  +  llama3.1:8b baseline')

# ════════════════════════ PHASE 4 — EVALUATION ═══════════════════════════════
Y, H = 4.65, 2.75
phase(0.3, Y, 14.4, H, C4, 4, 'Evaluation & Red Teaming', '92 cases · 10 metrics')

# 4 model chips (left)
rbox(0.6, Y+0.25, 2.3, H-1.0, INNER, C4, lw=1.5, r=0.15, z=4)
txt(1.75, Y+H-0.95, '4 models', size=9.5, color=C4, weight='bold', z=5)
for i, (m, c) in enumerate([('baseline', M_BASE), ('ft', M_FT), ('distill', M_DIST), ('curriculum', M_CURR)]):
    txt(1.75, Y+H-1.32-i*0.30, m, size=8.4, color=c, weight='bold', z=5)
arr_h(2.9, 3.45, Y+0.95)

ev = [('Automated Metrics', ('JSON · savings · budget', 'ROUGE-L · BERTScore', 'all-MiniLM-L6-v2')),
      ('LLM-as-Judge', ('DeepSeek V4 Flash', 'reasoning coherence', 'grounding accuracy')),
      ('Red Teaming · 45', ('adversarial prompts', 'budget overrides', 'injection attempts'))]
ew = 2.95
exs = [3.45, 6.55, 9.65]
for (name, lines), bx in zip(ev, exs):
    node(bx, Y+0.25, ew, H-1.0, C4, name, lines, tsize=10)
arr_h(exs[-1]+ew, 12.85, Y+0.95)
cylinder(13.55, Y+0.3, 2.1, H-1.1, C4, 'results', 'summary JSON + charts')

big_arrow(FW/2, Y, Y - 0.95, 'eval summary  +  4 charts')

# ════════════════════════ PHASE 5 — SERVING ══════════════════════════════════
Y, H = 0.55, 3.05
phase(0.3, Y, 14.4, H, C5, 5, 'FastAPI Inference Server', 'async · Pydantic · /docs')

node(0.6, Y+1.0, 3.0, 1.0, C5, 'FastAPI app',
     ('async httpx', 'Pydantic validation'))
arr_h(3.6, 4.15, Y+1.5, color=C5, lw=2.0)
node(4.15, Y+1.0, 2.9, 1.0, C5, 'Ollama runtime',
     ('local CPU', 'no GPU needed'))
arr_h(7.05, 7.6, Y+1.5, color=C5, lw=2.0)
node(7.6, Y+1.0, 3.0, 1.0, C5, '4 GGUF models',
     ('Q4_K_M · 4.6 GB', 'registry-validated'))

# Swagger badge
node(11.1, Y+1.0, 3.3, 1.0, C5, 'Swagger UI  /docs',
     ('auto-generated', 'OpenAPI schema'), fc=INNER2, tcolor='#7dd3fc', lw=2.2)

# endpoint chips row
eps = [('GET /health', C5), ('GET /models', C5), ('POST /optimize', '#7dd3fc'),
       ('GET /results/summary', C5), ('GET /results/compare', C5)]
ex = 0.85
for name, col in eps:
    w = 0.14*len(name) + 0.5
    chip(ex+w/2, Y+0.45, w, name, col)
    ex += w + 0.3

# ── Save ─────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / 'architecture.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG, pad_inches=0.3)
plt.close()
print(f'Saved: {out}')
