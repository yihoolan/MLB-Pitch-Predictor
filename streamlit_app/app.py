"""MLB Pitch Type Predictor — Streamlit dashboard.

Run from the project root (where the FastAPI server is also running):
    streamlit run streamlit_app/app.py

Expects the FastAPI server at http://localhost:8001 (configurable via API_URL).
"""

from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8001"

# ── Pitch type full names for display ─────────────────────────────────────────
PITCH_NAMES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FS": "Split-Finger",
    "KN": "Knuckleball",
    "ST": "Sweeper",
    "SV": "Slurve",
}

# Color families for the probability chart
PITCH_COLORS = {
    "FF": "#e63946",  # 4-seam — red
    "SI": "#f4a261",  # sinker — orange-red
    "FC": "#e9c46a",  # cutter — gold
    "SL": "#457b9d",  # slider — blue
    "CU": "#1d3557",  # curveball — dark blue
    "ST": "#6a4c93",  # sweeper — purple
    "SV": "#9d4edd",  # slurve — light purple
    "CH": "#2a9d8f",  # changeup — teal
    "FS": "#52b788",  # split-finger — green
    "KN": "#adb5bd",  # knuckleball — gray
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def search_players(query: str, role: str) -> list[dict]:
    if len(query.strip()) < 2:
        return []
    try:
        r = requests.get(f"{API_URL}/players", params={"query": query, "role": role}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_prediction(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Prediction failed: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Could not reach API: {e}")
        return None


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MLB Pitch Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚾ MLB Pitch Type Predictor")
st.caption("Predicts the type of the next pitch based on game situation and player matchup.")

# ── Session state defaults ────────────────────────────────────────────────────

for key, default in {
    "pitcher": None,       # selected PlayerMatch dict
    "batter": None,        # selected PlayerMatch dict
    "prediction": None,    # last PitchProbabilities response
    "whatif_balls": None,
    "whatif_strikes": None,
    "whatif_outs": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar: player search ────────────────────────────────────────────────────

with st.sidebar:
    st.header("Players")

    # Pitcher
    st.subheader("Pitcher")
    pitcher_query = st.text_input("Search pitcher name", placeholder="e.g. Gerrit Cole", key="pitcher_query")
    pitcher_results = search_players(pitcher_query, "pitcher")
    if pitcher_results:
        pitcher_options = {f"{p['name']}  (ID {p['mlbam_id']})": p for p in pitcher_results}
        selected_pitcher_label = st.selectbox("Select pitcher", list(pitcher_options.keys()), key="pitcher_select")
        st.session_state.pitcher = pitcher_options[selected_pitcher_label]
    elif pitcher_query and len(pitcher_query.strip()) >= 2:
        st.caption("No matches found.")

    if st.session_state.pitcher:
        st.success(f"✓ {st.session_state.pitcher['name']}")

    st.divider()

    # Batter
    st.subheader("Batter")
    batter_query = st.text_input("Search batter name", placeholder="e.g. Aaron Judge", key="batter_query")
    batter_results = search_players(batter_query, "batter")
    if batter_results:
        batter_options = {f"{p['name']}  (ID {p['mlbam_id']})": p for p in batter_results}
        selected_batter_label = st.selectbox("Select batter", list(batter_options.keys()), key="batter_select")
        st.session_state.batter = batter_options[selected_batter_label]
    elif batter_query and len(batter_query.strip()) >= 2:
        st.caption("No matches found.")

    if st.session_state.batter:
        st.success(f"✓ {st.session_state.batter['name']}")


# ── Main panel ────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Game Situation")

    # Handedness row
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        p_throws = st.radio("Pitcher throws", ["R", "L"], horizontal=True)
    with h_col2:
        stand = st.radio("Batter stands", ["R", "L"], horizontal=True)

    st.write("")

    # Count — balls, strikes, outs with progressive dot display
    c1, c2, c3 = st.columns(3)
    with c1:
        balls = st.select_slider("Balls", options=[0, 1, 2, 3], value=0)
        st.markdown("  ".join(["🔵"] * balls + ["⚪"] * (3 - balls)))
    with c2:
        strikes = st.select_slider("Strikes", options=[0, 1, 2], value=0)
        st.markdown("  ".join(["🔴"] * strikes + ["⚪"] * (2 - strikes)))
    with c3:
        outs = st.select_slider("Outs", options=[0, 1, 2], value=0)
        st.markdown("  ".join(["⬛"] * outs + ["⬜"] * (2 - outs)))

    st.write("")

    # Base runners — diamond layout
    st.write("**Runners on base**")
    d_top = st.columns([1, 1, 1])
    d_bot = st.columns([1, 1, 1])
    with d_top[1]:
        on_2b = st.checkbox("2B", key="on_2b")
    with d_bot[0]:
        on_3b = st.checkbox("3B", key="on_3b")
    with d_bot[2]:
        on_1b = st.checkbox("1B", key="on_1b")

    st.write("")

    # Inning / score / pitch number
    g1, g2, g3 = st.columns(3)
    with g1:
        inning = st.number_input("Inning", min_value=1, max_value=15, value=1, step=1)
    with g2:
        bat_score_diff = st.slider("Score diff", min_value=-10, max_value=10, value=0,
                                   help="Batting team score minus fielding team score")
    with g3:
        pitch_number = st.number_input("Pitch in AB", min_value=1, max_value=20, value=1, step=1)

    # Predict button
    st.write("")
    predict_disabled = st.session_state.pitcher is None or st.session_state.batter is None
    if predict_disabled:
        st.info("Select a pitcher and batter to predict.")

    if st.button("⚾  Predict Next Pitch", type="primary", disabled=predict_disabled, use_container_width=True):
        payload = {
            "pitcher_mlbam_id": st.session_state.pitcher["mlbam_id"],
            "pitcher_name": st.session_state.pitcher["name"],
            "batter_mlbam_id": st.session_state.batter["mlbam_id"],
            "batter_name": st.session_state.batter["name"],
            "balls": balls,
            "strikes": strikes,
            "outs_when_up": outs,
            "inning": inning,
            "pitch_number": pitch_number,
            "bat_score_diff": bat_score_diff,
            "on_1b": on_1b,
            "on_2b": on_2b,
            "on_3b": on_3b,
            "stand": stand,
            "p_throws": p_throws,
        }
        with st.spinner("Loading arsenal stats and predicting…"):
            result = get_prediction(payload)
        if result:
            st.session_state.prediction = result
            st.session_state.whatif_balls = balls
            st.session_state.whatif_strikes = strikes
            st.session_state.whatif_outs = outs
