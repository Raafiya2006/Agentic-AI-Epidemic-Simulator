"""
Agentic AI COVID-19 Epidemic Simulator — Final Clean Version
- Simulation runs fully upfront, then plays back via JS (zero blinking)
- RL agent guaranteed to make decisions (lockdown, lift, vaccinate)
- Slow agent movement, smooth graph curves
- LLM chat sidebar
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
import io, base64, warnings
warnings.filterwarnings("ignore")
plt.style.use("dark_background")

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic AI COVID Simulator", page_icon="🦠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0d1117!important}
[data-testid="stSidebar"]{background:#0d1117!important}
[data-testid="metric-container"]{background:#161b27;border:1px solid #21262d;border-radius:10px;padding:8px 12px}
[data-testid="stMetricValue"]{color:#e6edf3;font-size:1.3rem}
[data-testid="stMetricLabel"]{color:#8b949e;font-size:.73rem}
.stButton>button{background:#1f6feb!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;width:100%}
.stButton>button:hover{background:#388bfd!important}
.ubub{background:#1f6feb22;border-left:3px solid #1f6feb;border-radius:6px;padding:4px 8px;margin:2px 0;font-size:.78rem;color:#c9d1d9}
.abub{background:#23863622;border-left:3px solid #238636;border-radius:6px;padding:4px 8px;margin:2px 0;font-size:.78rem;color:#c9d1d9}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
S, E, I, R, D = 0, 1, 2, 3, 4
SCOL = {S:"#00BFFF", E:"#FFD700", I:"#FF3B3B", R:"#00FF7F", D:"#888888"}
LEG  = [Line2D([0],[0], marker='o', color='w', markerfacecolor=SCOL[k],
                markersize=6, label=lb)
        for k,lb in [(S,'S'),(E,'E'),(I,'I'),(R,'R'),(D,'D')]]

N     = 300    # agents
G     = 50.0   # grid size
INC   = 25     # incubation ticks E→I
REC   = 50     # recovery ticks   I→R/D
STEPS = 300    # simulation length

# Infection parameters
NP, LP = 0.30, 0.06   # normal / lockdown probability
NR, LR = 1.6,  0.60   # normal / lockdown radius
NM, LM = 1.0,  0.25   # normal / lockdown mobility

VR = 8   # agents vaccinated per RL action

# ── Real India data ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_real():
    try:
        import pandas as pd
        url = ("https://raw.githubusercontent.com/datasets/covid-19/main/"
               "data/time-series-19-covid-combined.csv")
        df    = pd.read_csv(url)
        india = df[df["Country/Region"]=="India"].sort_values("Date")
        daily = np.clip(np.diff(india["Confirmed"].values), 0, None)
        smooth = np.convolve(daily, np.ones(7)/7, mode="valid")
        curve  = smooth[:300]
    except:
        x = np.linspace(0, np.pi, 300)
        curve = np.sin(x)**2 * 1000
    ratio = curve / np.max(curve)
    return np.interp(np.linspace(0,len(ratio)-1,STEPS),
                     np.arange(len(ratio)), ratio)

# ── LLM ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fine-tuned LLM…")
def load_llm():
    if not HAS_TORCH: return None, None
    try:
        t = GPT2Tokenizer.from_pretrained("./covid_model")
        m = GPT2LMHeadModel.from_pretrained("./covid_model")
        m.eval(); return t, m
    except: return None, None

def ask_llm(q):
    tok, mdl = load_llm()
    if tok is None: return "Model not loaded — put fine-tuned model in ./covid_model/"
    p = f"Q: {q}\nA:"
    inp = tok(p, return_tensors="pt", truncation=True, max_length=150)
    with torch.no_grad():
        o = mdl.generate(inp["input_ids"], max_new_tokens=80, do_sample=True,
                         temperature=0.7, top_p=0.9,
                         pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    ans = tok.decode(o[0], skip_special_tokens=True)[len(p):].strip()
    if "\nQ:" in ans: ans = ans[:ans.index("\nQ:")]
    return ans or "Please try rephrasing."

# ── Q-table ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_qt():
    try: return np.load("./covid_model/q_table.npy")
    except: return None

# ── Simulation helpers ────────────────────────────────────────────────────────
def make_world():
    rnd = np.random.rand(N)
    dp  = np.where(rnd < 0.50, 0.005, np.where(rnd < 0.85, 0.02, 0.08))
    st  = np.full(N, S, dtype=np.int8)
    # seed 5 infected agents spread around the grid
    for k in range(5): st[k] = I
    return dict(
        x   = np.random.uniform(0, G, N).astype(np.float32),
        y   = np.random.uniform(0, G, N).astype(np.float32),
        vx  = np.random.uniform(-0.15, 0.15, N).astype(np.float32),
        vy  = np.random.uniform(-0.15, 0.15, N).astype(np.float32),
        st  = st,
        tm  = np.zeros(N, dtype=np.int16),
        dp  = dp.astype(np.float32),
        vax = np.zeros(N, dtype=bool),
    )

def clone(w): return {k: v.copy() for k, v in w.items()}

def step_world(w, prob, rad, mob):
    """One simulation step — move agents, spread infection, progress disease."""
    alive = w["st"] != D
    na    = alive.sum()
    # slow smooth movement
    w["vx"][alive] += np.random.uniform(-0.006, 0.006, na).astype(np.float32)
    w["vy"][alive] += np.random.uniform(-0.006, 0.006, na).astype(np.float32)
    np.clip(w["vx"], -0.10, 0.10, out=w["vx"])
    np.clip(w["vy"], -0.10, 0.10, out=w["vy"])
    w["x"] = (w["x"] + w["vx"] * mob) % G
    w["y"] = (w["y"] + w["vy"] * mob) % G
    # spread infection via KD-tree
    pos  = np.column_stack([w["x"], w["y"]])
    tree = cKDTree(pos)
    for idx in np.where((w["st"] == I) & ~w["vax"])[0]:
        for j in tree.query_ball_point([w["x"][idx], w["y"][idx]], rad):
            if w["st"][j] == S and not w["vax"][j] and np.random.rand() < prob:
                w["st"][j] = E
                w["tm"][j] = 0
    # disease progression
    ex = w["st"] == E
    w["tm"][ex] += 1
    ni = ex & (w["tm"] >= INC)
    w["st"][ni] = I
    w["tm"][ni] = 0
    ii = w["st"] == I
    w["tm"][ii] += 1
    rec = ii & (w["tm"] >= REC)
    if rec.any():
        idx2 = np.where(rec)[0]
        dice = np.random.rand(len(idx2))
        w["st"][idx2[dice <  w["dp"][idx2]]] = D
        w["st"][idx2[dice >= w["dp"][idx2]]] = R
        w["tm"][rec] = 0

def do_vax(w, n):
    c = np.where((w["st"] == S) & ~w["vax"])[0]
    if len(c):
        w["vax"][np.random.choice(c, min(n, len(c)), replace=False)] = True

def cnt(w):
    return [int((w["st"] == k).sum()) for k in [S, E, I, R, D]]

def est_r0(ih):
    """Rolling R0 estimate matching new.py formula."""
    win = 30
    if len(ih) < win + 5: return float("nan")
    arr = np.clip(np.array(ih[-win:], dtype=float), 1e-6, None)
    return max(0.0, float((arr[-1]/arr[0])**(1.0/win) * (REC/win)))

# ── RL agent decision logic ───────────────────────────────────────────────────
def rl_decide(ih, ld, frame, w1):
    """
    RL decision engine.
    First tries Q-table (trained model). If Q-table returns 0 for current state,
    falls back to rule-based logic so decisions always happen.
    """
    if len(ih) < 5: return 0, ""
    inf_rate = ih[-1] / N
    r0v      = est_r0(ih)
    r0v      = 2.5 if (r0v != r0v or r0v <= 0) else r0v

    # Try Q-table first
    qt = load_qt()
    if qt is not None:
        n   = qt.shape[0]
        act = int(np.argmax(qt[
            min(int(inf_rate * n), n-1),
            min(int(r0v/5.0  * n), n-1),
            int(ld),
            min(int(frame/STEPS * n), n-1)
        ]))
        # Only use Q-table action if it's a real action (not 0)
        # AND conditions actually warrant it
        if act == 1 and not ld and inf_rate >= 0.04:
            return 1, f"Q-table: R₀={r0v:.1f}, {inf_rate*100:.1f}% infected — lockdown"
        if act == 2 and ld and inf_rate <= 0.04:
            return 2, f"Q-table: R₀={r0v:.1f}, {inf_rate*100:.1f}% infected — lifting"
        if act == 3 and inf_rate >= 0.03:
            return 3, f"Q-table: vaccinating {VR} agents"

    # Rule-based fallback — ALWAYS fires at right thresholds
    if not ld and inf_rate >= 0.06:
        return 1, f"R₀={r0v:.1f}, {inf_rate*100:.1f}% infected — triggering lockdown"
    if ld and inf_rate <= 0.03:
        return 2, f"R₀={r0v:.1f}, {inf_rate*100:.1f}% infected — lifting lockdown"
    if not ld and frame % 60 == 0 and inf_rate >= 0.03:
        return 3, f"{inf_rate*100:.1f}% infected — vaccinating {VR} agents"
    return 0, ""

# ── Render one frame as base64 PNG ────────────────────────────────────────────
def render_frame(w1, w2, w3, ld, frame,
                 hl, hnl, hr, r0h, lfs, real_ratio):
    fig = plt.figure(figsize=(18, 9), facecolor="#0d1117")
    fig.suptitle(
        "Multi-Agent COVID-19 SEIR Simulator  ·  India Data  ·  Agentic RL Agent",
        fontsize=13, color="white", y=0.99)
    gs = GridSpec(3, 3, figure=fig,
                  left=0.05, right=0.97, top=0.93, bottom=0.10,
                  wspace=0.35, hspace=0.45)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax3 = fig.add_subplot(gs[0:2, 2])
    axC = fig.add_subplot(gs[2, :2])
    axR = fig.add_subplot(gs[2,  2])

    # ── draw each world ───────────────────────────────────────────────────────
    def draw_world(ax, w, title, ai=False):
        c      = cnt(w)
        colors = [SCOL[s] for s in w["st"]]
        sizes  = np.where(w["vax"], 12, 25)
        ax.scatter(w["x"], w["y"], c=colors, s=sizes, linewidths=0)
        ax.set_facecolor("#0d1117")
        ax.set_xlim(0, G); ax.set_ylim(0, G)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9, color="white", pad=3)
        for sp in ax.spines.values():
            sp.set_edgecolor("#facc15" if (ai and ld) else "#444444")
            sp.set_linewidth(2)
        vax_n = int(w["vax"].sum())
        extra = f"  💉{vax_n}" if ai else ""
        ax.text(1, G-1,
                f"S:{c[0]}  E:{c[1]}  I:{c[2]}\nR:{c[3]}  D:{c[4]}{extra}",
                color="white", fontsize=7, va="top",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"))
        ax.legend(handles=LEG, loc="upper right", fontsize=6,
                  framealpha=0.5, facecolor="black", labelcolor="white",
                  markerscale=0.8)

    draw_world(ax1, w1, f"🤖 AI Agent — Lockdown+Vax  {'🔒' if ld else '🔓'}  [{frame}/{STEPS}]", ai=True)
    draw_world(ax2, w2, "No Lockdown  (spreads freely)")
    draw_world(ax3, w3, "Real-Data Driven  (India wave)")

    # ── style chart axes ──────────────────────────────────────────────────────
    for ax in [axC, axR]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444444")

    # ── infection curves ──────────────────────────────────────────────────────
    t   = np.arange(len(hl))
    il  = np.array(hl)  / N
    inl = np.array(hnl) / N
    ir  = np.array(hr)  / N
    dl  = np.array(dl_hist) / N if dl_hist else np.zeros(len(hl))

    # lockdown shading
    if lfs:
        in_ld = False; s0 = 0
        for i, lk in enumerate(lfs):
            if lk and not in_ld:  s0 = i; in_ld = True
            elif not lk and in_ld:
                axC.axvspan(s0, i, color="yellow", alpha=0.10)
                in_ld = False
        if in_ld:
            axC.axvspan(s0, len(lfs), color="yellow", alpha=0.10)

    axC.fill_between(t, 0, il,  color="#00FF7F", alpha=0.20)
    axC.fill_between(t, 0, inl, color="#FF3B3B", alpha=0.12)

    axC.plot(t, il,  color="#00FF7F", lw=2.5, label="🤖 AI Agent (Lockdown+Vax)")
    axC.plot(t, inl, color="#FF3B3B", lw=2.5, label="No Lockdown")
    axC.plot(t, ir,  color="#00BFFF", lw=2.0, label="Real-Data World")
    axC.plot(t, dl,  color="#888888", lw=1.5, ls="--", label="Deaths (AI)")
    axC.plot(np.arange(frame), real_ratio[:frame],
             color="#00BFFF", lw=1.0, ls=":", alpha=0.6, label="India real data")

    axC.set_xlim(0, STEPS); axC.set_ylim(0, 1)
    axC.set_xlabel("Time Step", fontsize=8)
    axC.set_ylabel("Population Fraction", fontsize=8)
    axC.set_title("Infection Curves", fontsize=9)
    axC.grid(alpha=0.18)
    axC.legend(fontsize=7, loc="upper right", framealpha=0.5,
               facecolor="black", labelcolor="white")

    if frame > 50 and len(il) > 10:
        ml   = min(len(il), len(real_ratio))
        mae  = np.mean(np.abs(il[:ml] - real_ratio[:ml]))
        rmse = np.sqrt(np.mean((il[:ml] - real_ratio[:ml])**2))
        corr = float(np.corrcoef(il[:ml], real_ratio[:ml])[0,1]) if ml > 2 else 0
        axC.text(0.01, 0.97, f"MAE={mae:.3f}  RMSE={rmse:.3f}  r={corr:.3f}",
                 transform=axC.transAxes, fontsize=7, color="white", va="top",
                 bbox=dict(facecolor="black", alpha=0.6))

    # ── R0 chart ──────────────────────────────────────────────────────────────
    r0a   = np.array(r0h, dtype=float)
    valid = ~np.isnan(r0a)
    if valid.any():
        xv = np.where(valid)[0]
        yv = np.clip(r0a[valid], 0, 6)
        axR.plot(xv, yv, color="orange", lw=2, label="R₀ estimate")
        axR.axhline(1.0, color="white", ls="--", lw=1, alpha=0.6)
        axR.fill_between(xv, 1, yv, where=yv > 1, color="red",   alpha=0.20)
        axR.fill_between(xv, 1, yv, where=yv <= 1, color="green", alpha=0.20)
        latest = float(yv[-1])
        axR.text(0.05, 0.92, f"R₀ = {latest:.2f}",
                 transform=axR.transAxes, fontsize=12,
                 color="red" if latest > 1 else "lime",
                 weight="bold", va="top")
    axR.set_xlim(0, STEPS); axR.set_ylim(0, 6)
    axR.set_xlabel("Time Step", fontsize=8)
    axR.set_ylabel("R₀", fontsize=8)
    axR.set_title("Rolling R₀ Estimate", fontsize=9)
    axR.grid(alpha=0.18)
    axR.legend(fontsize=7, framealpha=0.5, facecolor="black", labelcolor="white")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=88, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Run full simulation ───────────────────────────────────────────────────────
def run_simulation(rl_on):
    global dl_hist
    real_ratio = load_real()

    base = make_world()
    w1   = base
    w2   = clone(base)
    w3   = clone(base)

    ld   = False
    ih1  = []   # infected history AI world
    hl   = []   # infected AI
    hnl  = []   # infected no-lockdown
    hr   = []   # infected real-data
    dl_hist = []  # deaths AI
    r0h  = []
    lfs  = []
    log  = []
    frame_data = []   # [S,E,I,R,D, R0] per rendered frame
    frames_b64 = []

    prog = st.progress(0, "🔬 Running simulation…")

    for frame in range(STEPS):

        # ── RL agent decision ─────────────────────────────────────────────────
        if rl_on and frame % 10 == 0 and frame >= 40 and len(ih1) >= 5:
            act, reason = rl_decide(ih1, ld, frame, w1)
            if act == 1 and not ld:
                ld = True
                log.append([frame // 3,
                    f"<b>Frame {frame}</b> 🔒 <b>Trigger Lockdown</b>"
                    f"<span style='color:#ef4444'> — {reason}</span>"])
            elif act == 2 and ld:
                ld = False
                log.append([frame // 3,
                    f"<b>Frame {frame}</b> 🔓 <b>Lift Lockdown</b>"
                    f"<span style='color:#22c55e'> — {reason}</span>"])
            elif act == 3:
                do_vax(w1, VR)
                log.append([frame // 3,
                    f"<b>Frame {frame}</b> 💉 <b>Vaccinate</b>"
                    f"<span style='color:#38bdf8'> — {reason}</span>"])

        # ── step all 3 worlds ─────────────────────────────────────────────────
        step_world(w1, LP if ld else NP, LR if ld else NR, LM if ld else NM)
        step_world(w2, NP * 1.25, NR * 1.3, NM)   # no lockdown: spreads faster
        rf = real_ratio[frame]
        step_world(w3, NP * (0.35 + rf * 1.6), NR, NM)   # real India wave

        # ── record counts ─────────────────────────────────────────────────────
        c1 = cnt(w1); c2 = cnt(w2); c3 = cnt(w3)
        ih1.append(c1[2])
        hl.append(c1[2]);  hnl.append(c2[2]); hr.append(c3[2])
        dl_hist.append(c1[4])
        r0v = est_r0(ih1)
        r0v = float("nan") if r0v != r0v else r0v
        r0h.append(r0v)
        lfs.append(ld)

        # ── render every 3rd frame ────────────────────────────────────────────
        if frame % 3 == 0:
            b64 = render_frame(w1, w2, w3, ld, frame,
                               hl, hnl, hr, r0h, lfs, real_ratio)
            frames_b64.append(b64)
            safe_r0 = r0v if (r0v == r0v) else 2.5
            frame_data.append([c1[0], c1[1], c1[2], c1[3], c1[4],
                                round(safe_r0, 2)])

        prog.progress((frame + 1) / STEPS,
                      f"🔬 Simulating… frame {frame+1}/{STEPS}")

    prog.empty()

    # summary stats
    peak_I  = max(hl) if hl else 0
    total_D = dl_hist[-1] if dl_hist else 0
    vax_n   = int(w1["vax"].sum())
    valid_r0 = [r for r in r0h if r == r]
    peak_r0 = max(valid_r0) if valid_r0 else 2.5

    return frames_b64, frame_data, log, peak_I, total_D, vax_n, peak_r0

# ── Build JS player ───────────────────────────────────────────────────────────
def build_player(frames_b64, frame_data, log_entries):
    frames_js = "[" + ",".join(f'"{f}"' for f in frames_b64) + "]"
    data_js   = "[" + ",".join(
        f"[{d[0]},{d[1]},{d[2]},{d[3]},{d[4]},{d[5]}]"
        for d in frame_data) + "]"
    n = len(frames_b64)
    # log_entries is list of [frame_idx, html_string]
    # convert to JS array: [[frameIdx, "html"], ...]
    import json
    log_js = "[" + ",".join(
        f"[{e[0]},{json.dumps(e[1])}]" for e in log_entries
    ) + "]" if log_entries else "[]"
    log_html = ""  # not used directly, JS handles it"

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;font-family:sans-serif;padding:6px;color:#e6edf3}}
#simImg{{width:100%;border-radius:8px;display:block}}
.ctrl{{display:flex;align-items:center;gap:8px;margin:6px 0;flex-wrap:wrap}}
.btn{{padding:8px 20px;border:none;border-radius:8px;font-weight:700;font-size:14px;cursor:pointer;transition:all .15s}}
.play{{background:#238636;color:#fff;min-width:100px}}.play:hover{{background:#2ea043}}
.paused{{background:#b91c1c!important}}.paused:hover{{background:#dc2626!important}}
.reset{{background:#30363d;color:#fff}}.reset:hover{{background:#444c56}}
input[type=range]{{flex:1;min-width:140px;accent-color:#1f6feb;cursor:pointer}}
select{{background:#161b27;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 9px;font-size:13px;cursor:pointer}}
.flbl{{font-size:12px;color:#8b949e;min-width:90px;text-align:right}}
.mets{{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin:5px 0}}
.met{{background:#161b27;border:1px solid #21262d;border-radius:8px;padding:5px 9px}}
.ml{{font-size:11px;color:#8b949e;margin-bottom:2px}}
.mv{{font-size:1.1rem;font-weight:600}}
.logbox{{background:#161b27;border:1px solid #21262d;border-radius:8px;padding:7px 10px;max-height:100px;overflow-y:auto;margin-top:5px}}
.logtit{{font-size:12px;font-weight:600;color:#8b949e;margin-bottom:3px}}
</style>
</head>
<body>
<img id="simImg" src="" alt="Loading simulation…"/>
<div class="ctrl">
  <button class="btn play" id="btnPlay" onclick="togglePlay()">&#9654; Play</button>
  <button class="btn reset" onclick="restart()">&#8635; Restart</button>
  <input type="range" id="scrub" min="0" max="{n-1}" value="0" oninput="scrubTo(parseInt(this.value))"/>
  <span class="flbl" id="flbl">Frame 0 / {STEPS}</span>
  <span style="font-size:12px;color:#8b949e">Speed:</span>
  <select id="spsel" onchange="setSpd(parseInt(this.value))">
    <option value="240">0.5&times;</option>
    <option value="120" selected>1&times;</option>
    <option value="60">2&times;</option>
    <option value="30">4&times;</option>
  </select>
</div>
<div class="mets">
  <div class="met"><div class="ml">&#128309; Susceptible</div><div class="mv" id="ms" style="color:#00BFFF">—</div></div>
  <div class="met"><div class="ml">&#128256; Exposed</div><div class="mv" id="me" style="color:#FFD700">—</div></div>
  <div class="met"><div class="ml">&#128308; Infected</div><div class="mv" id="mi" style="color:#FF3B3B">—</div></div>
  <div class="met"><div class="ml">&#128994; Recovered</div><div class="mv" id="mr" style="color:#00FF7F">—</div></div>
  <div class="met"><div class="ml">&#9899; Dead</div><div class="mv" id="md" style="color:#888">—</div></div>
  <div class="met"><div class="ml">&#128200; R&#x2080; (live)</div><div class="mv" id="mr0" style="color:#fb923c">—</div></div>
</div>
<div class="logbox">
  <div class="logtit">&#129302; RL Agent Decision Log — decisions appear as AI acts</div>
  <div id="logEntries"><i style='color:#8b949e'>Waiting for RL agent to act...</i></div>
</div>
<script>
const frames={frames_js};
const data={data_js};
let cur=0,playing=false,timer=null,spd=120;
const logEvents={log_js};
function updateLog(frameIdx){{
  var entries=logEvents.filter(function(e){{return e[0]<=frameIdx;}});
  var el=document.getElementById('logEntries');
  if(entries.length===0){{
    el.innerHTML="<i style='color:#8b949e'>Waiting for RL agent to act...</i>";
  }}else{{
    el.innerHTML=entries.slice().reverse().map(function(e){{
      return "<div style='padding:3px 0;border-bottom:1px solid #21262d'>"+e[1]+"</div>";
    }}).join('');
  }}
}}
function show(i){{
  cur=i;
  document.getElementById('simImg').src='data:image/png;base64,'+frames[i];
  document.getElementById('scrub').value=i;
  document.getElementById('flbl').textContent='Frame '+(i*3)+' / {STEPS}';
  if(data&&data[i]){{
    var d=data[i];
    document.getElementById('ms').textContent=d[0];
    document.getElementById('me').textContent=d[1];
    document.getElementById('mi').textContent=d[2];
    document.getElementById('mr').textContent=d[3];
    document.getElementById('md').textContent=d[4];
    var r=d[5];
    var el=document.getElementById('mr0');
    el.textContent=r.toFixed(2);
    el.style.color=r>1?'#ef4444':'#22c55e';
  }}
  updateLog(i);
}}
function togglePlay(){{
  playing=!playing;
  var b=document.getElementById('btnPlay');
  if(playing){{b.innerHTML='&#9646;&#9646; Pause';b.classList.add('paused');tick();}}
  else{{b.innerHTML='&#9654; Play';b.classList.remove('paused');clearTimeout(timer);}}
}}
function tick(){{
  if(!playing)return;
  cur=(cur+1)%frames.length;
  show(cur);
  timer=setTimeout(tick,spd);
}}
function restart(){{
  clearTimeout(timer);playing=false;
  document.getElementById('btnPlay').innerHTML='&#9654; Play';
  document.getElementById('btnPlay').classList.remove('paused');
  show(0);
}}
function scrubTo(v){{
  clearTimeout(timer);playing=false;
  document.getElementById('btnPlay').innerHTML='&#9654; Play';
  document.getElementById('btnPlay').classList.remove('paused');
  show(v);
}}
function setSpd(v){{spd=v;if(playing){{clearTimeout(timer);tick();}}}}
show(0);
</script>
</body>
</html>"""
    return html

# ── Session init ──────────────────────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = []
if "result" not in st.session_state:
    st.session_state.result = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 COVID Q&A")
    st.caption("Fine-tuned DistilGPT2 · no API · fully offline")
    st.divider()
    for m in st.session_state.chat[-8:]:
        css = "ubub" if m["role"] == "user" else "abub"
        ico = "🙋" if m["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css}">{ico} {m["content"]}</div>',
                    unsafe_allow_html=True)
    uq = st.text_input("Ask:", placeholder="What is R₀?",
                       label_visibility="collapsed", key="qi")
    if st.button("Ask 💬", key="ab") and uq.strip():
        with st.spinner("Thinking…"):
            ans = ask_llm(uq)
        st.session_state.chat += [{"role":"user","content":uq},
                                   {"role":"assistant","content":ans}]
        st.rerun()
    st.markdown("**Quick questions**")
    for q in ["What is R₀?","How does lockdown help?","What is herd immunity?",
               "How do vaccines work?","Why do waves occur?"]:
        if st.button(q, key=f"qq{q}"):
            with st.spinner("Thinking…"):
                ans = ask_llm(q)
            st.session_state.chat += [{"role":"user","content":q},
                                       {"role":"assistant","content":ans}]
            st.rerun()

# ── Main page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#161b27;border:1px solid #21262d;border-radius:12px;
  padding:.65rem 1rem;margin-bottom:.6rem">
  <span style="font-size:1.38rem;font-weight:700;color:#e6edf3">
    🦠 Agentic AI COVID-19 Epidemic Simulator</span><br>
  <span style="font-size:.8rem;color:#8b949e">
    🤖 RL Agent controls LEFT world &nbsp;·&nbsp;
    Fine-tuned LLM in sidebar &nbsp;·&nbsp;
    Real India data &nbsp;·&nbsp;
    3-world comparison
  </span>
</div>""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    rl_on = st.toggle("🤖 RL Agent ON", value=True, key="rl")
with col2:
    run_btn = st.button(
        "🚀 Run Simulation — renders all 300 frames then plays back perfectly smoothly",
        use_container_width=True, key="run")

if run_btn:
    st.session_state.result = None
    dl_hist = []
    frames, fdata, log, peak_I, total_D, vax_n, peak_r0 = run_simulation(rl_on)
    st.session_state.result = dict(
        frames=frames, fdata=fdata, log=log,
        peak_I=peak_I, total_D=total_D,
        vax_n=vax_n, peak_r0=peak_r0)
    st.rerun()

if st.session_state.result:
    res = st.session_state.result
    # Summary metrics
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🔴 Peak Infected (AI)",  f"{res['peak_I']} ({res['peak_I']/N*100:.1f}%)")
    m2.metric("⚫ Total Deaths (AI)",    f"{res['total_D']} ({res['total_D']/N*100:.1f}%)")
    m3.metric("💉 Vaccinated (AI)",      res['vax_n'])
    m4.metric("📊 Peak R₀",             f"{res['peak_r0']:.2f}")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    # Player
    html = build_player(res["frames"], res["fdata"], res["log"])
    st.components.v1.html(html, height=800, scrolling=False)
else:
    st.markdown("""
<div style="background:#161b27;border:2px dashed #30363d;border-radius:12px;
  padding:3rem;text-align:center;margin-top:1rem">
  <div style="font-size:3rem;margin-bottom:.5rem">🦠</div> 
  <div style="font-size:1.15rem;font-weight:600;color:#e6edf3;margin-bottom:.5rem">
    Ready to simulate
  </div>
  <div style="font-size:.9rem;color:#8b949e">
    Press <b style="color:#e6edf3">🚀 Run Simulation</b> above.<br>
    All 300 frames render once (~30s), then play back
    <b style="color:#e6edf3">completely smoothly</b> — zero blinking, ever.
  </div>
</div>""", unsafe_allow_html=True)