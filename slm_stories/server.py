"""
Gradio chat UI for the trained SLM story generator.
Dark, minimalist, ChatGPT-style interface.

Run:
    python server.py
Then open http://localhost:7860
"""

from __future__ import annotations

import threading
from dataclasses import fields
from pathlib import Path
from typing import Iterator, Optional

import tiktoken
import torch
import torch.nn.functional as F
import gradio as gr

from config import SLMConfig, config
from model import SLM


SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH  = SCRIPT_DIR / config.checkpoint_dir / "best.pt"

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
_model:  Optional[SLM]               = None
_enc:    Optional[tiktoken.Encoding]  = None
_cfg:    Optional[SLMConfig]          = None
_device: Optional[torch.device]       = None
_lock = threading.Lock()


def _cfg_from_ckpt(saved: dict) -> SLMConfig:
    cfg = SLMConfig()
    known = {f.name for f in fields(SLMConfig)}
    for k, v in saved.items():
        if k in known:
            setattr(cfg, k, v)
    return cfg


def load_model():
    global _model, _enc, _cfg, _device
    if _model is not None:
        return _model, _enc, _cfg, _device
    with _lock:
        if _model is not None:
            return _model, _enc, _cfg, _device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}. Run train.py first.")
        ckpt  = torch.load(CKPT_PATH, map_location=device)
        cfg   = _cfg_from_ckpt(ckpt.get("config", {}))
        model = SLM(cfg).to(device)
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()}
        model.load_state_dict(state)
        model.eval()
        enc   = tiktoken.get_encoding("gpt2")
        _model, _enc, _cfg, _device = model, enc, cfg, device
        val    = ckpt.get("val_loss", 0.0)
        itr    = ckpt.get("iter", "?")
        params = model.get_num_params() / 1e6
        print(f"[OK] Loaded  iter={itr}  val_loss={val:.4f}  params={params:.1f}M  device={device}")
    return _model, _enc, _cfg, _device


# ---------------------------------------------------------------------------
# Core generation — yields the growing output string token by token
# ---------------------------------------------------------------------------
def _stream(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> Iterator[str]:
    model, enc, cfg, device = load_model()

    ids = enc.encode_ordinary(prompt) if prompt.strip() else [cfg.eot_token]
    if len(ids) > cfg.block_size:
        ids = ids[-cfg.block_size:]

    head_dim = cfg.n_embd // cfg.n_heads
    dtype    = model.tok_emb.weight.dtype

    def empty():
        z = torch.zeros(1, cfg.n_heads, 0, head_dim, dtype=dtype, device=device)
        return z, z.clone()

    caches = [empty() for _ in model.blocks]
    idx    = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    T, x   = idx.size(1), None

    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with torch.no_grad(), autocast:
        for t in range(T):
            tok   = idx[:, t:t+1]
            pos_t = torch.tensor([[t]], dtype=torch.long, device=device)
            x     = model.tok_emb(tok) + model.pos_emb(pos_t)
            nc    = []
            for i, blk in enumerate(model.blocks):
                x, c = blk(x, kv_cache=caches[i])
                nc.append(c)
            caches = nc

        output = prompt
        for step in range(max_new_tokens):
            if T + step >= cfg.block_size:
                break
            logits = model.lm_head(model.norm_f(x)[:, -1:, :]).squeeze(1)
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            tok_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            if tok_id == cfg.eot_token:
                break
            output += enc.decode([tok_id])
            yield output

            pos_t = torch.tensor([[T + step]], dtype=torch.long, device=device)
            x     = model.tok_emb(torch.tensor([[tok_id]], device=device)) + model.pos_emb(pos_t)
            nc    = []
            for i, blk in enumerate(model.blocks):
                x, c = blk(x, kv_cache=caches[i])
                nc.append(c)
            caches = nc


# ChatInterface fn: receives message str + history list, yields reply str
def chat_fn(message: str, history: list,
            max_tokens: int, temperature: float, top_k: int) -> Iterator[str]:
    if not message.strip():
        yield "Please enter a story prompt."
        return
    for partial in _stream(message, max_tokens, temperature, top_k):
        # yield only the new part (without the echoed prompt)
        yield partial[len(message):]


# ---------------------------------------------------------------------------
# CSS — dark, minimal, GPT-like
# ---------------------------------------------------------------------------
CSS = """
:root {
    --bg:       #212121;
    --surface:  #2f2f2f;
    --surface2: #3a3a3a;
    --border:   #444444;
    --accent:   #10a37f;
    --accent-h: #0d8a6b;
    --text:     #ececec;
    --muted:    #8e8e8e;
    --radius:   14px;
}

/* ── page shell ── */
body, .gradio-container, .main, .wrap {
    background: var(--bg) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.gradio-container { max-width: 800px !important; margin: 0 auto !important; padding: 0 16px !important; }
footer, .built-with, .svelte-byatnx { display: none !important; }

/* ── title block ── */
#slm-title {
    text-align: center;
    padding: 28px 0 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
#slm-title h1 { color: var(--text); font-size: 1.5rem; font-weight: 600; margin: 0; letter-spacing: -0.3px; }
#slm-title p  { color: var(--muted); font-size: 0.82rem; margin: 5px 0 10px; }
.pill {
    display: inline-block;
    background: var(--surface2);
    color: var(--accent);
    border: 1px solid #2e5f4f;
    border-radius: 999px;
    font-size: 0.7rem;
    padding: 3px 12px;
    font-weight: 500;
    letter-spacing: 0.4px;
}

/* ── chatbot ── */
.chatbot { background: var(--bg) !important; border: none !important; }

/* user bubble */
.message.user { justify-content: flex-end !important; }
.message.user .bubble-wrap {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-radius: var(--radius) var(--radius) 4px var(--radius) !important;
    padding: 10px 15px !important;
    max-width: 75% !important;
    font-size: 0.95rem !important;
}

/* bot bubble */
.message.bot .bubble-wrap {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) var(--radius) var(--radius) 4px !important;
    padding: 14px 18px !important;
    max-width: 90% !important;
    font-size: 0.97rem !important;
    line-height: 1.8 !important;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* ── textbox / input area ── */
.gr-textbox, label > textarea, textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.95rem !important;
    caret-color: var(--accent) !important;
}
textarea:focus { outline: none !important; border-color: var(--accent) !important; }
textarea::placeholder { color: var(--muted) !important; }

/* ── send / stop buttons ── */
button[aria-label="Submit"], #submit-btn, button.primary {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: background 0.15s !important;
}
button[aria-label="Submit"]:hover, #submit-btn:hover, button.primary:hover {
    background: var(--accent-h) !important;
}
button[aria-label="Stop"], button.stop {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    transition: all 0.15s !important;
}
button[aria-label="Stop"]:hover, button.stop:hover {
    border-color: #e55 !important;
    color: #e55 !important;
}

/* ── settings accordion ── */
.gr-accordion { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
.gr-accordion .label-wrap span { color: var(--muted) !important; font-size: 0.82rem !important; }
input[type=range] { accent-color: var(--accent) !important; }
label span { color: var(--muted) !important; font-size: 0.8rem !important; }

/* ── example chips ── */
.gr-sample-textbox, .gr-samples-table button {
    background: var(--surface) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    font-size: 0.78rem !important;
    padding: 4px 12px !important;
    transition: all 0.15s !important;
}
.gr-sample-textbox:hover, .gr-samples-table button:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
}
"""

HEADER_HTML = """
<div id="slm-title">
  <h1>SLM Story Generator</h1>
  <p>51M-parameter transformer &mdash; trained on TinyStories &mdash; RTX 5060 8GB</p>
  <span class="pill">val loss 1.64 &nbsp;&middot;&nbsp; bfloat16 &nbsp;&middot;&nbsp; 5k iters</span>
</div>
"""

# [prompt, max_tokens, temperature, top_k]
EXAMPLES = [
    ["Once upon a time there was a little girl named Lily.",        300, 0.8, 40],
    ["One day a boy named Tom found a red ball in the park.",       250, 0.9, 50],
    ["In the forest, there lived a friendly bear who loved honey.", 300, 0.8, 40],
    ["The dragon was sad because nobody wanted to be friends.",     350, 0.85, 40],
    ["Mia loved to paint but one day she ran out of all her colors.", 300, 0.8, 40],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="SLM Story Generator",
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.emerald,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        additional = [
            gr.Slider(50,  500, value=300, step=10,    label="Max tokens"),
            gr.Slider(0.1, 1.5, value=0.8, step=0.05,  label="Temperature"),
            gr.Slider(1,   100, value=40,  step=1,      label="Top-k"),
        ]

        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=additional,
            additional_inputs_accordion=gr.Accordion("Settings", open=False),
            chatbot=gr.Chatbot(height=460, render_markdown=False, show_label=False),
            textbox=gr.Textbox(
                placeholder="Type a story opener and press Enter...",
                show_label=False,
                lines=1,
                max_lines=5,
                autofocus=True,
            ),
            examples=EXAMPLES,
            title=None,
            description=None,
            submit_btn="Send",
            stop_btn="Stop",
            autofocus=True,
        )

    return demo


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    ui = build_ui()
    ui.queue()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
