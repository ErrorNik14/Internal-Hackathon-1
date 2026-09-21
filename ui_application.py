"""Fantasy XI recommender.  Run with:  streamlit run ui_application.py"""
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import train as T
import evaluate as E

st.set_page_config(page_title="Fantasy XI", layout="wide")

DEFAULT_PATHS = ("src/model artifacts/idata_20220326_20251231.nc",
                 "src/model artifacts/meta_20220326_20251231.pkl")


@st.cache_resource
def get_artifacts(idata_path, meta_path):
    return E.load_artifacts(idata_path, meta_path)


def current_artifacts():
    idata_path, meta_path = st.session_state.get("paths", DEFAULT_PATHS)
    if not (Path(idata_path).exists() and Path(meta_path).exists()):
        return None
    return get_artifacts(str(idata_path), str(meta_path))


st.title("Dream XI PS")
product_tab, model_tab = st.tabs(["Recommend a team", "Evaluate the model"])


# --------------------------------------------------------------------------
with product_tab:
    teams = sorted(set(T.data['batting_team']) | set(T.data['bowling_team']))
    c1, c2, c3 = st.columns(3)
    team_1 = c1.selectbox("Team 1", teams, index=0)
    team_2 = c2.selectbox("Team 2", teams, index=1)
    match_date = c3.date_input("Match date", value=date(2026, 4, 12))

    if st.button("Recommend XI", type="primary"):
        if team_1 == team_2:
            st.error("Pick two different teams.")
        else:
            art = current_artifacts()
            if art is None:
                st.error("No trained model found. Train one in the other tab first.")
                st.stop()
            t0 = time.perf_counter()
            try:
                xi = E.recommend_xi(team_1, team_2, match_date, art)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            elapsed = time.perf_counter() - t0

            if elapsed > 10:
                st.warning(f"Took {elapsed:.1f}s, over the 10 second limit.")

            roles = xi["role"].value_counts().to_dict()
            split = xi["team"].value_counts().to_dict()
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected points", f"{xi['pred_points'].sum():.0f}")
            m2.metric("Roles", ", ".join(f"{v} {k}" for k, v in roles.items()))
            m3.metric("Team split", " / ".join(str(v) for v in split.values()))

            st.dataframe(
                xi[["name", "role", "team", "pred_points", "p_over_50"]].rename(columns={
                    "name": "Player", "role": "Role", "team": "Team",
                    "pred_points": "Expected points", "p_over_50": "Chance of 50+"}),
                hide_index=True, use_container_width=True)

            st.subheader("Why these players")
            for r in xi.itertuples(index=False):
                with st.expander(f"{r.name}, {r.pred_points:.1f} points"):
                    st.write(r.justification)

            st.caption(f"Computed in {elapsed:.2f}s")


# --------------------------------------------------------------------------
with model_tab:
    st.subheader("Train")
    c1, c2 = st.columns(2)
    train_start = c1.date_input("Train from", value=date(2022, 3, 26))
    train_end = c2.date_input("Train to", value=date(2025, 12, 31))

    if st.button("Retrain model"):
        try:
            with st.spinner("Training"):
                idata_path, meta_path, diag = T.train(train_start, train_end)
            st.session_state["paths"] = (str(idata_path), str(meta_path))
            st.success(f"Saved to {idata_path}")
            st.write(f"R-hat max {diag['r_hat_max']:.2f} · "
                     f"min ESS {diag['ess_min']:.0f} · "
                     f"divergences {diag['divergences']}")
        except ValueError as e:
            st.error(str(e))

    st.divider()
    st.subheader("Evaluate")
    c1, c2 = st.columns(2)
    test_start = c1.date_input("Test from", value=date(2026, 3, 28))
    test_end = c2.date_input("Test to", value=date(2026, 5, 31))

    if st.button("Run evaluation", type="primary"):
        art = current_artifacts()
        if art is None:
            st.error("No trained model found. Train one above first.")
            st.stop()
        with st.spinner("Predicting every match in the window"):
            metrics, res = E.evaluate(test_start, test_end, art)

        if metrics["matches"] == 0:
            st.info("No matches in that window. Widen the date range.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean overlap",
                      f"{metrics['mean_overlap']:.2f} / 11",
                      delta=f"{metrics['mean_overlap'] - metrics['random_overlap']:+.2f} vs random")
            c2.metric("Mean capture",
                      f"{metrics['mean_capture_pct']:.1f}%",
                      delta=f"{metrics['mean_capture_pct'] - metrics['random_capture_pct']:+.1f} vs random")
            c3.metric("Matches", metrics["matches"])

            c4, c5, c6 = st.columns(3)
            c4.metric("Player MAE", f"{metrics['player_mae']:.2f}",
                      delta=f"{metrics['player_mae'] - metrics['player_mae_median_baseline']:+.2f} vs median",
                      delta_color="inverse")
            c5.metric("10–90 coverage", f"{metrics['coverage_10_90']:.2f}",
                      help="Share of real scores inside the predicted 10–90 range. Target 0.80.")
            c6.metric("Random overlap", f"{metrics['random_overlap']:.2f} / 11")

            st.caption(f"Standard error: ±{metrics['overlap_se']:.2f} overlap, "
                       f"±{metrics['capture_se']:.1f}% capture")

            st.dataframe(res, hide_index=True, use_container_width=True)
            st.download_button("Download CSV", res.to_csv(index=False).encode(),
                               file_name=f"predictions_{test_start}_{test_end}.csv",
                               mime="text/csv")