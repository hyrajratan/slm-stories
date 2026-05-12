import streamlit as st
import torch
import tiktoken
from dataclasses import fields

REPO_ID = "hyrajratan/slm-stories"

st.set_page_config(page_title="SLM Story Generator", page_icon="📖", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    from huggingface_hub import hf_hub_download
    from config import SLMConfig
    from model import SLM

    ckpt = torch.load(hf_hub_download(REPO_ID, "best.pt"), map_location="cpu")
    cfg = SLMConfig()
    for k, v in ckpt.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = SLM(cfg)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()})
    model.eval()
    return model, tiktoken.get_encoding("gpt2"), cfg

# ── initialise prompt in session state ───────────────────────────────────────
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📖 SLM Story Generator")
st.caption("A small language model trained on TinyStories — type a prompt and watch it write!")

st.markdown("### Try a prompt")
examples = [
    "Once upon a time there was a little girl named Lily.",
    "One day a boy named Tom found a red ball.",
    "In the forest, there lived a friendly bear.",
    "The little dog wanted to make a new friend.",
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(f'"{ex[:38]}…"', use_container_width=True, key=f"ex{i}"):
        st.session_state.prompt = ex   # ← update BEFORE text_area renders

prompt = st.text_area("Your prompt", value=st.session_state.prompt, height=80, placeholder="Start your story here…")

if st.button("✨ Generate story", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Loading model… (first run only)"):
            model, enc, cfg = load_model()
        from generate import generate_story
        with st.spinner("Writing your story…"):
            story = generate_story(model, enc, cfg, torch.device("cpu"), prompt.strip(),
                                   max_new_tokens=200, temperature=0.8, top_k=40)
        st.markdown("---")
        st.markdown("### 📝 Generated story")
        st.markdown(story)
        st.download_button("⬇️ Download as .txt", story, "story.txt")

st.markdown("---")
st.caption("Built with [Streamlit](https://streamlit.io) · Model on [Hugging Face](https://huggingface.co) · Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)")