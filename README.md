# Fantasy XI Predictor & Recommender (Dream11 IPL Model)

A Streamlit app that recommends an optimal Fantasy (Dream11-style) Playing XI for an
upcoming IPL match, using a model trained on historical ball-by-ball delivery data.
The app also lets you retrain the model, switch between saved models, and evaluate
model accuracy against actual 2026-season results.

## What the app does

The UI has two tabs:

- **Product UI** — pick two IPL teams and a match date, click **Recommend XI**, and
  get:
  - A recommended 11-player fantasy team with each player's role (BAT / BOWL /
    AR / WK), real team, and predicted fantasy points.
  - Summary metrics: total expected points, role breakdown, and team split.
  - An expandable "Player Insights" section with a short justification for each
    pick.
- **Model UI** — for anyone who wants to look under the hood:
  - **Retrain** a new Linear Regression model from the delivery datasets over a
    chosen date range, and save it as a new `.pkl` file.
  - **Select** which saved model (`.pkl` file in `Stat_Features/`) is currently
    active for predictions.
  - **Evaluate** the active model on real 2026-season matches (predicted vs.
    actual fantasy points, mean absolute error, players correctly picked out of
    11), and download the results as CSV.

## Prerequisites

- Python 3.10–3.12
- ~2–3 GB free disk space (the datasets and dependencies, including `jax`/`numpyro`,
  are sizeable)
- No internet access is needed at runtime — everything runs on local data/model
  files bundled in this repo.

## 1. Clone the repo and set up an environment

```bash
git clone https://github.com/ErrorNik14/Internal-Hackathon-1/
cd Internal-Hackathon-1-main

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, pandas/numpy/scikit-learn, and a few extras used
elsewhere in the project (PyMC, NumPyro, JAX, PuLP) that aren't required to run
the UI itself but are listed for the full modeling pipeline. If you only want to
run the app and installation of those heavier packages is a problem on your
machine, it's safe to comment them out of `requirements.txt` — the Streamlit app
does not import them.


## 3. Run the app

```bash
streamlit run ui_app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) and should
open it automatically in your browser. If not, open that URL manually.

   - Number of matches evaluated
5. Results are saved automatically to `results/predictions_<start>_<end>.csv` and
   can also be downloaded via the **Download Evaluation CSV** button.


