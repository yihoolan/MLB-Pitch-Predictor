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


@st.cache_data(ttl=3600)
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


def _render_prediction_chart(result: dict, title_suffix: str = "") -> None:
    """Render a horizontal bar chart of pitch-type probabilities."""
    import matplotlib.pyplot as plt

    probs = result["probabilities"]
    # Sort by probability descending; drop pitches below 1%
    items = sorted(probs.items(), key=lambda x: -x[1])
    items = [(pt, p) for pt, p in items if p >= 0.01]

    pitch_labels = [f"{PITCH_NAMES.get(pt, pt)} ({pt})" for pt, _ in items]
    values = [p * 100 for _, p in items]
    colors = [PITCH_COLORS.get(pt, "#adb5bd") for pt, _ in items]

    fig, ax = plt.subplots(figsize=(6, max(3, len(items) * 0.55)))
    bars = ax.barh(pitch_labels[::-1], values[::-1], color=colors[::-1], edgecolor="none", height=0.6)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=9
        )

    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()

    top = result["top_pitch"]
    top_name = PITCH_NAMES.get(top, top)
    header = f"Most likely: **{top_name} ({top})** — {probs[top] * 100:.1f}%{title_suffix}"
    st.markdown(header)

    if result.get("rookie_pitcher") or result.get("rookie_batter"):
        names = []
        if result.get("rookie_pitcher"):
            names.append(result["pitcher_name"])
        if result.get("rookie_batter"):
            names.append(result["batter_name"])
        st.caption(f"⚠️ No prior-year stats found for {', '.join(names)} — using league average.")

    st.pyplot(fig)
    plt.close(fig)


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
    "pitcher": None,  # selected PlayerMatch dict
    "batter": None,  # selected PlayerMatch dict
    "prediction": None,  # last PitchProbabilities response
    "whatif_balls": None,
    "whatif_strikes": None,
    "whatif_outs": None,
    "pitcher_id_prev": None,
    "batter_id_prev": None,
    "p_throws": "R",
    "stand": "R",
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
    if len(pitcher_results) == 1:
        st.session_state.pitcher = pitcher_results[0]
    elif len(pitcher_results) > 1:
        pitcher_options = {p["name"]: p for p in pitcher_results}
        selected = st.selectbox(
            "Select pitcher", list(pitcher_options.keys()),
            index=None, placeholder="Choose a pitcher…", key="pitcher_select",
        )
        if selected:
            st.session_state.pitcher = pitcher_options[selected]
    elif pitcher_query and len(pitcher_query.strip()) >= 2:
        st.caption("No matches found.")
    if st.session_state.pitcher:
        st.success(f"✓ {st.session_state.pitcher['name']}")

    st.divider()

    # Batter
    st.subheader("Batter")
    batter_query = st.text_input("Search batter name", placeholder="e.g. Aaron Judge", key="batter_query")
    batter_results = search_players(batter_query, "batter")
    if len(batter_results) == 1:
        st.session_state.batter = batter_results[0]
    elif len(batter_results) > 1:
        batter_options = {p["name"]: p for p in batter_results}
        selected = st.selectbox(
            "Select batter", list(batter_options.keys()),
            index=None, placeholder="Choose a batter…", key="batter_select",
        )
        if selected:
            st.session_state.batter = batter_options[selected]
    elif batter_query and len(batter_query.strip()) >= 2:
        st.caption("No matches found.")
    if st.session_state.batter:
        st.success(f"✓ {st.session_state.batter['name']}")


# ── Sync handedness when selected player changes ──────────────────────────────

pitcher_id = st.session_state.pitcher["mlbam_id"] if st.session_state.pitcher else None
if pitcher_id != st.session_state.pitcher_id_prev:
    if st.session_state.pitcher:
        hand = st.session_state.pitcher.get("throws_or_stands", "?")
        if hand in ("L", "R"):
            st.session_state.p_throws = hand
    st.session_state.pitcher_id_prev = pitcher_id

batter_id = st.session_state.batter["mlbam_id"] if st.session_state.batter else None
if batter_id != st.session_state.batter_id_prev:
    if st.session_state.batter:
        hand = st.session_state.batter.get("throws_or_stands", "?")
        if hand in ("L", "R"):
            st.session_state.stand = hand
    st.session_state.batter_id_prev = batter_id


# ── Main panel ────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Game Situation")

    # Handedness row
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        p_throws = st.radio("Pitcher throws", ["R", "L"], horizontal=True, key="p_throws")
    with h_col2:
        stand = st.radio("Batter stands", ["R", "L"], horizontal=True, key="stand")

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
        bat_score_diff = st.slider(
            "Score diff", min_value=-10, max_value=10, value=0, help="Batting team score minus fielding team score"
        )
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


# ── Right column: probability chart ──────────────────────────────────────────

with col_right:
    st.subheader("Prediction")

    if st.session_state.prediction is None:
        st.info("Fill in the game situation and click **Predict Next Pitch**.")
    else:
        _render_prediction_chart(st.session_state.prediction)


# ── What-If explorer ──────────────────────────────────────────────────────────

if st.session_state.prediction is not None:
    with st.expander("What-If: Change the Count", expanded=False):
        st.caption("Adjust the count to see how the pitch distribution changes vs. your original prediction.")

        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            wi_balls = st.select_slider(
                "Balls ", options=[0, 1, 2, 3], value=st.session_state.whatif_balls, key="wi_balls"
            )
            st.markdown("  ".join(["🔵"] * wi_balls + ["⚪"] * (3 - wi_balls)))
        with wf2:
            wi_strikes = st.select_slider(
                "Strikes ", options=[0, 1, 2], value=st.session_state.whatif_strikes, key="wi_strikes"
            )
            st.markdown("  ".join(["🔴"] * wi_strikes + ["⚪"] * (2 - wi_strikes)))
        with wf3:
            wi_outs = st.select_slider("Outs ", options=[0, 1, 2], value=st.session_state.whatif_outs, key="wi_outs")
            st.markdown("  ".join(["⬛"] * wi_outs + ["⬜"] * (2 - wi_outs)))

        count_changed = (
            wi_balls != st.session_state.whatif_balls
            or wi_strikes != st.session_state.whatif_strikes
            or wi_outs != st.session_state.whatif_outs
        )

        if count_changed:
            orig = st.session_state.prediction
            wi_payload = {
                "pitcher_mlbam_id": st.session_state.pitcher["mlbam_id"],
                "pitcher_name": st.session_state.pitcher["name"],
                "batter_mlbam_id": st.session_state.batter["mlbam_id"],
                "batter_name": st.session_state.batter["name"],
                "balls": wi_balls,
                "strikes": wi_strikes,
                "outs_when_up": wi_outs,
                "inning": orig.get("inning", inning),
                "pitch_number": orig.get("pitch_number", pitch_number),
                "bat_score_diff": orig.get("bat_score_diff", bat_score_diff),
                "on_1b": on_1b,
                "on_2b": on_2b,
                "on_3b": on_3b,
                "stand": stand,
                "p_throws": p_throws,
            }
            with st.spinner("Updating…"):
                wi_result = get_prediction(wi_payload)

            if wi_result:
                ch1, ch2 = st.columns(2)
                orig_count = (
                    f"Original  "
                    f"({st.session_state.whatif_balls}-{st.session_state.whatif_strikes}, "
                    f"{st.session_state.whatif_outs} out)"
                )
                wi_count = f"What-If  ({wi_balls}-{wi_strikes}, {wi_outs} out)"
                with ch1:
                    st.markdown(f"**{orig_count}**")
                    _render_prediction_chart(orig)
                with ch2:
                    st.markdown(f"**{wi_count}**")
                    _render_prediction_chart(wi_result)
