import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from daytona_sdk import Daytona, DaytonaConfig

load_dotenv()

ASSETS = Path(__file__).parent / "assets"

from models import DemoState
from sandbox import run_sandbox

TASK_SCRIPT = Path(__file__).parent / "cartpole_task.py"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daytona · CartPole RL Demo",
    page_icon=Image.open(ASSETS / "main_daytona_logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp { background-color: #0D0D0D; color: #E0E0E0; }
.stApp * { font-family: 'Inter', 'Helvetica Neue', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #00C8FF22;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2 { color: #00C8FF; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #161616;
    border: 1px solid #00C8FF22;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] {
    color: #00C8FF;
    font-size: 1.7rem !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #666;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Buttons ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button {
    background-color: #00C8FF;
    color: #000;
    border: none;
    font-weight: 700;
    border-radius: 6px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover { background-color: #00A8D8; }
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent;
    color: #FF4444;
    border: 1px solid #FF444466;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background-color: #FF444411;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div { background-color: #00C8FF !important; }
[data-testid="stProgressBar"] > div { background-color: #1A1A1A !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #00C8FF22;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stDataFrame"] table { background-color: #111111 !important; }
[data-testid="stDataFrame"] th {
    background-color: #0D0D0D !important;
    color: #00C8FF !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #00C8FF33 !important;
}
[data-testid="stDataFrame"] td { color: #CCC !important; }

/* ── Divider ── */
hr { border-color: #222 !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background-color: #1A1A1A !important;
    color: #E0E0E0 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
}
.stSlider [data-testid="stSliderTrack"] { background-color: #00C8FF !important; }

/* ── Status badges ── */
.badge-running  { color: #00C8FF; font-weight: 700; }
.badge-complete { color: #00FF88; font-weight: 700; }
.badge-idle     { color: #555;    font-weight: 700; }

/* ── Page title ── */
.page-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #00C8FF;
    letter-spacing: 0.06em;
    margin-bottom: 0.1rem;
}
.page-sub {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
def _init():
    defaults = {
        "state":          None,
        "thread":         None,
        "running":        False,
        "daytona_client": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
ss = st.session_state


# ── Background worker ──────────────────────────────────────────────────────────
def _run_demo(state, client, n, episodes, workers, batch_size, batch_delay, lrs, task_code):
    indices = list(range(n))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for batch_start in range(0, n, batch_size):
            batch = indices[batch_start:batch_start + batch_size]
            for i in batch:
                futures[executor.submit(
                    run_sandbox, i, state, client, episodes, lrs[i], task_code
                )] = i
            if batch_start + batch_size < n:
                time.sleep(batch_delay)
        for f in futures:
            try:
                f.result()
            except Exception:
                pass


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_df(state: DemoState) -> pd.DataFrame:
    status_label = {
        "pending":  "... pending",
        "running":  "> running",
        "complete": "+ done",
        "error":    "x error",
    }
    with state.lock:
        rows = [
            {
                "#":          str(r.index + 1),
                "Sandbox ID": (r.sandbox_id[:14] + "...") if len(r.sandbox_id) > 14 else r.sandbox_id,
                "Status":     status_label.get(r.status, r.status),
                "Episode":    str(r.episode)  if r.episode  else "-",
                "Avg(100)":   str(r.avg_100)  if r.avg_100  else "-",
                "Best":       str(r.best)     if r.best     else "-",
                "Solved":     "yes" if r.solved else "-",
                "LR":         str(r.lr)       if r.lr       else "-",
            }
            for r in sorted(state.results.values(), key=lambda r: r.index)
        ]
    return pd.DataFrame(rows)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(str(ASSETS / "main_daytona_logo.png"), width='stretch')
    st.markdown('<p style="color:#555; font-size:0.82rem; margin-top:-8px; margin-bottom:12px;">CartPole-v1 · Distributed RL</p>', unsafe_allow_html=True)

    api_key = st.text_input(
        "API Key",
        value=os.environ.get("DAYTONA_API_KEY", ""),
        type="password",
        placeholder="dtn_...",
    )

    st.divider()
    st.markdown('<p style="color:#888; font-size:0.78rem; text-transform:uppercase; letter-spacing:.08em;">Run Config</p>', unsafe_allow_html=True)

    n_sandboxes = st.number_input("Sandboxes",   min_value=1,   max_value=1000, value=25,  step=1)
    episodes    = st.number_input("Episodes",    min_value=10,  max_value=1000, value=100, step=10)
    batch_size  = st.number_input("Batch size",  min_value=1,   max_value=200,  value=25,  step=5)
    batch_delay = st.slider("Batch delay (s)", min_value=1.0, max_value=30.0, value=3.0, step=0.5)

    st.divider()

    run_col, kill_col = st.columns(2)
    with run_col:
        if st.button("Run", disabled=ss.running or not api_key, width="stretch"):
            task_code  = TASK_SCRIPT.read_text()
            lr_choices = [0.001, 0.005, 0.01, 0.02, 0.05]
            lrs        = [random.choice(lr_choices) for _ in range(n_sandboxes)]
            workers    = min(n_sandboxes, 200)
            client     = Daytona(DaytonaConfig(api_key=api_key))

            ss.daytona_client = client
            ss.state          = DemoState(total=n_sandboxes)
            ss.running        = True

            t = threading.Thread(
                target=_run_demo,
                args=(ss.state, client, n_sandboxes, episodes, workers,
                      batch_size, batch_delay, lrs, task_code),
                daemon=True,
            )
            ss.thread = t
            t.start()
            st.rerun()

    with kill_col:
        if st.button("Kill All", width="stretch", type="secondary"):
            if api_key:
                with st.spinner("Deleting..."):
                    client = Daytona(DaytonaConfig(api_key=api_key))
                    result = client.list()
                    for s in result.items:
                        try:
                            client.delete(s)
                        except Exception:
                            pass
                st.success(f"Deleted {result.total} sandbox(es).")


# ── Main area ──────────────────────────────────────────────────────────────────
title_col, logo_col = st.columns([5, 1])
with title_col:
    st.markdown('<div class="page-title">Daytona Infra · CartPole-v1 Distributed RL</div>', unsafe_allow_html=True)
with logo_col:
    st.image(str(ASSETS / "main_daytona_logo.png"), width='stretch')

# Check if background thread finished
if ss.running and ss.thread and not ss.thread.is_alive():
    ss.running = False

# Status badge
if ss.state is None:
    st.markdown('<span class="badge-idle">● IDLE</span>&nbsp; Configure a run in the sidebar and click Run.', unsafe_allow_html=True)
    st.divider()
    hero_l, hero_r = st.columns(2)
    with hero_l:
        st.image(str(ASSETS / "blue_isolated_container.png"), caption="Isolated sandboxes, spun up in parallel")
    with hero_r:
        st.image(str(ASSETS / "daytona_speed_timer.png"), caption="90ms sandbox creation")
    inf_l, inf_c, inf_r = st.columns([1, 2, 1])
    with inf_c:
        st.image(str(ASSETS / "green_infinity_rectangle.png"), caption="Scale to thousands of sandboxes")
elif ss.running:
    st.markdown(f'<span class="badge-running">● RUNNING</span>&nbsp; {ss.state.elapsed}s elapsed', unsafe_allow_html=True)
else:
    st.markdown(f'<span class="badge-complete">● COMPLETE</span>&nbsp; finished in {ss.state.elapsed}s', unsafe_allow_html=True)

if ss.state is not None:
    state  = ss.state
    n      = state.total
    n_done = state.n_complete
    n_run  = state.n_running
    n_err  = state.n_error

    st.divider()

    # ── Metrics ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Running",   n_run)
    c2.metric("Complete",  f"{n_done} / {n}")
    c3.metric("Failed",    n_err)
    c4.metric("Solved",    state.n_solved)
    c5.metric("Solve %",   f"{round(state.n_solved / max(n_done, 1) * 100, 1)}%")
    c6.metric("Avg Score", state.avg_final)

    # ── Progress bar ──
    st.progress(n_done / n if n > 0 else 0)

    st.divider()

    # ── Results table ──
    df = _build_df(state)
    if not df.empty:
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=min(40 + len(df) * 35, 600),
            column_config={
                "#":         st.column_config.NumberColumn(width="small"),
                "Sandbox ID":st.column_config.TextColumn(width="medium"),
                "Status":    st.column_config.TextColumn(width="medium"),
                "Episode":   st.column_config.Column(width="small"),
                "Avg(100)":  st.column_config.Column(width="small"),
                "Best":      st.column_config.Column(width="small"),
                "Solved":    st.column_config.TextColumn(width="small"),
                "LR":        st.column_config.Column(width="small"),
            },
        )

# ── Live polling ───────────────────────────────────────────────────────────────
if ss.running:
    time.sleep(0.5)
    st.rerun()
