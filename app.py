import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load models
# -----------------------------
score_model = joblib.load("t20i_first_innings_score_model.joblib")
win_model   = joblib.load("t20i_second_innings_winprob_model.joblib")

BALLS_PER_INNINGS = 120

st.set_page_config(page_title="T20I Cricket Predictor", layout="centered")

st.title("🏏 T20I Cricket Prediction App")

tab1, tab2 = st.tabs(["1️⃣ First Innings Score", "2️⃣ Second Innings Win Probability"])

# =====================================================
# TAB 1: FIRST INNINGS SCORE PREDICTION
# =====================================================
with tab1:
    st.subheader("First Innings Final Score Prediction")

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.text_input("Batting Team", key="fi_batting_team")
        bowling_team = st.text_input("Bowling Team", key="fi_bowling_team")
        venue = st.text_input("Venue", key="fi_venue")

    with col2:
        overs = st.number_input("Overs Bowled", min_value=0, max_value=19, value=10, key="fi_overs")
        balls = st.number_input("Balls in Current Over", min_value=0, max_value=5, value=0, key="fi_balls")

        current_score = st.number_input("Current Score", min_value=0, value=80, key="fi_current_score")
        wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=3, key="fi_wickets")

    runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, value=40, key="fi_runs_last_5")
    wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, value=1, key = "fi_wickets_last_5")

    balls_bowled = overs * 6 + balls
    crr = (current_score / balls_bowled) * 6 if balls_bowled > 0 else 0

    if st.button("Predict Final Score", key="fi_predict_btn"):
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "current_score": current_score,
            "balls_bowled": balls_bowled,
            "wickets_fallen": wickets_fallen,
            "runs_last_5": runs_last_5,
            "wickets_last_5": wickets_last_5,
            "crr": crr
        }])

        prediction = score_model.predict(input_df)[0]
        st.success(f"🏏 Predicted Final Score: **{int(round(prediction))} runs**")

# =====================================================
# TAB 2: SECOND INNINGS WIN PROBABILITY
# =====================================================
with tab2:
    st.subheader("Second Innings Win Probability")

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.text_input("Chasing Team", key = "si_batting_team")
        bowling_team = st.text_input("Bowling Team", key = "si_bowling_team")
        venue = st.text_input("Venue ", key = "si_venue")

    with col2:
        target = st.number_input("Target", min_value=1, value=160, key = "si_target")
        overs = st.number_input("Overs Bowled ", min_value=0, max_value=19, value=12, key = "si_overs")
        balls = st.number_input("Balls in Current Over ", min_value=0, max_value=5, value=0, key = "si_balls")

        current_score = st.number_input("Current Score ", min_value=0, value=95, key = "si_current_score")
        wickets_fallen = st.number_input("Wickets Fallen ", min_value=0, max_value=9, value=4, key = "si_wickets")

    runs_last_5 = st.number_input("Runs in Last 5 Overs ", min_value=0, value=35, key = "si_runs_last_5")
    wickets_last_5 = st.number_input("Wickets in Last 5 Overs ", min_value=0, value=1, key = "si_wickets_last_5")

    balls_bowled = overs * 6 + balls
    balls_left = BALLS_PER_INNINGS - balls_bowled
    runs_left = target - current_score
    wickets_left = 10 - wickets_fallen

    crr = (current_score / balls_bowled) * 6 if balls_bowled > 0 else 0
    rrr = (runs_left / balls_left) * 6 if balls_left > 0 else 99

    if st.button("Predict Win Probability", key = "si_predict_btn"):
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "target": target,
            "current_score": current_score,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "crr": crr,
            "rrr": rrr,
            "runs_last_5": runs_last_5,
            "wickets_last_5": wickets_last_5
        }])

        win_prob = win_model.predict_proba(input_df)[0][1]

        st.success(f"🎯 Win Probability: **{win_prob*100:.2f}%**")
        st.progress(int(win_prob * 100))

