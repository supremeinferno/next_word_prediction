import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="WordFlow · Next Word AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #05070f !important;
    color: #cbd5e1;
}
.stApp { background: #05070f !important; }

#MainMenu, header, footer { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

/* Kill ALL default Streamlit spacing */
.block-container {
    padding: 0 2rem 5rem !important;
    max-width: 780px !important;
}
.element-container { margin: 0 !important; padding: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* ── Keyframes ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glowPulse {
    0%, 100% { opacity: 0.55; transform: translateX(-50%) scale(1); }
    50%       { opacity: 0.85; transform: translateX(-50%) scale(1.08); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes chipIn {
    from { opacity: 0; transform: translateY(14px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes barGrow {
    from { opacity: 0; transform: scaleX(0); }
    to   { opacity: 1; transform: scaleX(1); }
}
@keyframes borderGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
    50%       { box-shadow: 0 0 20px 4px rgba(99,102,241,0.18); }
}

/* ── Background glow ── */
.stApp::before {
    content: '';
    position: fixed;
    top: -280px; left: 50%;
    transform: translateX(-50%);
    width: 900px; height: 900px;
    background: radial-gradient(circle, rgba(99,102,241,0.14) 0%, rgba(168,85,247,0.06) 40%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: glowPulse 6s ease-in-out infinite;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -200px; right: -150px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(168,85,247,0.09) 0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
}

/* ── Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 0 1.5rem;
    animation: fadeUp 0.5s ease both;
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.nav-badge {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 100px;
    padding: 0.32rem 0.85rem;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 1.5rem 0 2.8rem;
    animation: fadeUp 0.6s 0.1s ease both;
}
.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5.5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #f1f5f9;
    margin-bottom: 1.1rem;
}
.hero-title .grad {
    background: linear-gradient(135deg, #818cf8, #a78bfa, #e879f9, #818cf8);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
}
.hero-sub {
    font-size: 1.02rem;
    font-weight: 400;
    color: #64748b;
    line-height: 1.75;
}

/* ── Input card — pure CSS, no wrapping st.markdown div ── */
.icard {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px 20px 0 0;
    padding: 1.4rem 1.6rem 1rem;
    margin-bottom: 0;
}
.card-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.7rem;
}
.icard-bottom {
    padding: 0.5rem 1.6rem 1.6rem;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    border-top: none;
    border-radius: 0 0 20px 20px;
    margin-top: 0;
}
            
/* ── Textarea ── */

[data-testid="stTextArea"] > div > div {
    border-radius: 0 !important;
    border-left: none !important;
    border-right: none !important;
    border-color: rgba(255,255,255,0.09) !important;
    background: rgba(255,255,255,0.035) !important;
    margin: 0 !important;
}
            
textarea {
    border-radius: 0 !important;  
    border: none !important;
    background: transparent !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
    caret-color: #818cf8 !important;
    resize: none !important;
    padding: 1rem 1.15rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
textarea:focus {
    border-color: rgba(99,102,241,0.55) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 0 40px rgba(99,102,241,0.07) !important;
    outline: none !important;
}
textarea::placeholder { color: #334155 !important; }

/* Remove label gap that Streamlit injects */
[data-testid="stTextArea"] { margin-top: 0 !important; }
[data-testid="stTextArea"] label { display: none !important; }
[data-testid="stTextArea"] > div { margin-top: 0 !important; }

/* ── Slider ── */
[data-testid="stSlider"] {
    padding-top: 0.6rem !important;
}
[data-testid="stSlider"] > label {
    font-size: 0.67rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #818cf8 !important;
    border-color: #a5b4fc !important;
}

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.85rem !important;
    cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(79,70,229,0.38) !important;
    transition: box-shadow 0.25s, transform 0.2s, opacity 0.2s !important;
    margin-top: 0.8rem !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 36px rgba(124,58,237,0.55) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    opacity: 0.9 !important;
}

/* ── Section header ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2.2rem 0 1.3rem;
    animation: fadeUp 0.5s ease both;
}
.sec-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #64748b;
    white-space: nowrap;
}
.sec-line { flex: 1; height: 1px; background: rgba(255,255,255,0.07); }

/* ── Preview box ── */
.preview-box {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    font-size: 1.15rem;
    line-height: 1.7;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    font-weight: 300;
    animation: fadeUp 0.45s ease both;
}
.preview-box .pw {
    color: #c4b5fd;
    font-weight: 700;
    border-bottom: 2px solid rgba(196,181,253,0.45);
    padding-bottom: 2px;
    font-size: 1.18rem;
}

/* ── Word chips ── */
.chips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(145px, 1fr));
    gap: 0.8rem;
    margin-bottom: 2rem;
}
.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1.05rem 1.15rem;
    display: flex;
    flex-direction: column;
    gap: 0.38rem;
    cursor: default;
    transition: border-color 0.2s, background 0.2s, transform 0.18s, box-shadow 0.2s;
    opacity: 0;
    animation: chipIn 0.4s ease forwards;
}
.chip:nth-child(1) { animation-delay: 0.05s; }
.chip:nth-child(2) { animation-delay: 0.10s; }
.chip:nth-child(3) { animation-delay: 0.15s; }
.chip:nth-child(4) { animation-delay: 0.20s; }
.chip:nth-child(5) { animation-delay: 0.25s; }
.chip:nth-child(6) { animation-delay: 0.30s; }
.chip:nth-child(7) { animation-delay: 0.35s; }
.chip:nth-child(8) { animation-delay: 0.40s; }
.chip:nth-child(9) { animation-delay: 0.45s; }
.chip:nth-child(10){ animation-delay: 0.50s; }
.chip:hover {
    border-color: rgba(129,140,248,0.4);
    background: rgba(99,102,241,0.09);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.15);
}
.chip.top {
    border-color: rgba(129,140,248,0.45);
    background: rgba(99,102,241,0.1);
}
.chip-rank {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #818cf8;
}
.chip-word {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
}
.chip.top .chip-word { color: #c4b5fd; }
.chip-conf {
    font-size: 0.7rem;
    color: #64748b;
    font-weight: 500;
}

/* ── Confidence bars ── */
.bar-word {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    line-height: 1;
}
.bar-pct {
    font-size: 0.75rem;
    color: #64748b;
    text-align: right;
    line-height: 1;
}
[data-testid="stProgress"] {
    padding: 0 !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 100px !important;
    height: 6px !important;
    overflow: hidden;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
    border-radius: 100px !important;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* align bar rows vertically */
.bar-row-cols > div {
    display: flex !important;
    align-items: center !important;
}

/* ── Footer ── */
.site-footer {
    font-size: 0.68rem;
    color: #1e293b;
    text-align: center;
    margin-top: 4.5rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model     = load_model("lstm_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    max_len   = pickle.load(open("max_len.pkl", "rb"))
    return model, tokenizer, max_len

def predict_next_words(model, tokenizer, text, max_len, top_n=5):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
    probs      = model.predict(token_list, verbose=0)[0]
    top_ids    = np.argsort(probs)[::-1][:top_n]
    idx2word   = {v: k for k, v in tokenizer.word_index.items()}
    results    = []
    for rank, idx in enumerate(top_ids, 1):
        word = idx2word.get(idx, "")
        if word:
            results.append({"rank": rank, "word": word, "prob": float(probs[idx])})
    return results


# ══════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════

# Navbar
st.markdown("""
<div class="navbar">
  <span class="nav-logo">⚡ WordFlow</span>
  <span class="nav-badge">LSTM · Deep Learning</span>
</div>""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Language Model</div>
  <div class="hero-title">Predict what comes <span class="grad">next</span></div>
  <div class="hero-sub">Type a phrase and let the LSTM model<br>complete your thought in real time.</div>
</div>""", unsafe_allow_html=True)

# Load model silently
with st.spinner("Warming up..."):
    try:
        model, tokenizer, max_len = load_artifacts()
    except Exception as e:
        st.error(f"Could not load model files — make sure `lstm_model.h5`, `tokenizer.pkl` and `max_len.pkl` are in the same folder.\n\n`{e}`")
        st.stop()

# ── Input — top half of card ──
st.markdown("""
<div class="icard">
  <div class="card-label">Your prompt</div>
</div>""", unsafe_allow_html=True)

# Streamlit widget placed directly — no extra div wrapper
user_input = st.text_area(
    label="prompt",
    placeholder='e.g.  "The universe is full of"',
    height=115,
    label_visibility="collapsed",
)

# Bottom half of card (slider + button share the card look)
st.markdown('<div class="icard-bottom">', unsafe_allow_html=True)
top_n = st.slider("Suggestions", min_value=1, max_value=10, value=5)
predict_btn = st.button("Generate predictions →", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("Enter a phrase above first.", icon="💬")
    else:
        with st.spinner("Running inference..."):
            preds = predict_next_words(model, tokenizer, user_input.strip(), max_len, top_n)

        if not preds:
            st.error("No predictions — some words may be out of vocabulary.", icon="🔍")
        else:
            top_word = preds[0]["word"]

            # Preview
            st.markdown("""
            <div class="sec-header">
              <span class="sec-label">Predictions</span>
              <div class="sec-line"></div>
            </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div class="preview-box">{user_input.strip()} '
                f'<span class="pw">{top_word}</span></div>',
                unsafe_allow_html=True,
            )

            # Chips
            chips_html = '<div class="chips-grid">'
            for p in preds:
                cls   = "chip top" if p["rank"] == 1 else "chip"
                label = "#1 · top pick" if p["rank"] == 1 else f"#{p['rank']}"
                chips_html += (
                    f'<div class="{cls}">'
                    f'<span class="chip-rank">{label}</span>'
                    f'<span class="chip-word">{p["word"]}</span>'
                    f'<span class="chip-conf">{p["prob"]*100:.2f}% confidence</span>'
                    f'</div>'
                )
            chips_html += "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

            # Confidence bars
            st.markdown("""
            <div class="sec-header">
              <span class="sec-label">Confidence breakdown</span>
              <div class="sec-line"></div>
            </div>""", unsafe_allow_html=True)

            for p in preds:
                c1, c2, c3 = st.columns([2, 6, 1])
                with c1:
                    st.markdown(f'<div class="bar-word">{p["word"]}</div>', unsafe_allow_html=True)
                with c2:
                    st.progress(float(p["prob"]))
                with c3:
                    st.markdown(f'<div class="bar-pct">{p["prob"]*100:.1f}%</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="site-footer">Built by SupremeInferno · Powered by LSTM</div>', unsafe_allow_html=True)




#python3 -m streamlit run app.py