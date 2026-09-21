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
git clone <this-repo-url>
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

## 3. Check the data files are present

Everything the app needs is already included in this repo — you shouldn't need to
download anything extra. Confirm the following exist relative to the project
root before launching:

```
Datasets/IPL DELIVERIES/ipl_2022_deliveries.csv
Datasets/IPL DELIVERIES/ipl_2023_deliveries.csv
Datasets/IPL DELIVERIES/ipl_2024_deliveries.csv
Datasets/IPL DELIVERIES/ipl_2025_deliveries.csv
Datasets/IPL DELIVERIES/ipl_2026_deliveries.csv
Datasets/ipl_json/                      # per-match Cricsheet JSON files, used for the "Evaluate" feature
csv files/player_behaviour_data.csv     # used to infer player roles (BAT/BOWL/AR/WK)
csv files/player_roles.csv              # fallback role mapping
Stat_Features/Dream11_Team_Regressor.pkl  # the default pre-trained model
```

> **Important:** always run the `streamlit run` command from the project's root
> folder (the one containing `Datasets/`, `csv files/`, and `Stat_Features/`).
> The app loads all data using paths relative to the current working directory,
> so running it from anywhere else will show "file not found" errors in the UI.

## 4. Run the app

```bash
streamlit run ui_app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) and should
open it automatically in your browser. If not, open that URL manually.

## 5. Using the UI

### Recommend a Fantasy XI (Product UI tab)

1. Open the **Product UI** tab (selected by default).
2. Choose **Team 1** and **Team 2** from the dropdowns (must be two different
   teams).
3. Pick the **Match Date** for the fixture you're predicting.
4. Click **Recommend XI**.
5. Review the recommended 11 players, the expected points table, and expand any
   player's row under "Player Insights" to see the reasoning behind their
   inclusion.

The recommendation should complete in a few seconds; if it takes longer than 10
seconds the app will show a timing warning (this doesn't affect the result).

### Switch or retrain the model (Model UI tab)

- **Retrain:** set a model name, a training start date, and a training end date,
  then click **Start Retraining Pipeline**. The new model is saved into
  `Stat_Features/` and automatically becomes the active model.
- **Select an existing model:** pick any `.pkl` file listed under
  `Stat_Features/` from the dropdown and click **Set Active Model**.
- You can also type a custom path directly into the **Active Model Path** field
  in the left sidebar (useful if you've copied a model file in from elsewhere).

### Evaluate the active model

1. In the **Model UI** tab, scroll to **Evaluate Active Model on Season 2026**.
2. Choose a **Test From** / **Test To** date range.
3. Click **Run Evaluation**.
4. The app compares the model's predicted XI against the actual match results
   for that period and reports:
   - Average number of players correctly matched (out of 11)
   - Mean Absolute Error (MAE) on predicted fantasy points
   - Number of matches evaluated
5. Results are saved automatically to `results/predictions_<start>_<end>.csv` and
   can also be downloaded via the **Download Evaluation CSV** button.

## Project structure

```
Internal-Hackathon-1-main/
├── ui_app.py                          # Streamlit app (run this)
├── player_role_generator.py           # Standalone script that derives player_behaviour_data.csv / player_roles.csv
├── Dream11_Hackathon_Statistical_Model.ipynb  # Notebook used for exploration / model development
├── requirements.txt
├── Datasets/
│   ├── IPL DELIVERIES/                # Ball-by-ball delivery CSVs (2022–2026), primary model input
│   └── ipl_json/                      # Cricsheet match JSON files, used for name-matching & evaluation
├── csv files/
│   ├── player_behaviour_data.csv      # Batting/bowling participation % per player (drives role inference)
│   └── player_roles.csv               # Precomputed role fallback
├── Stat_Features/
│   ├── Dream11_Team_Regressor.pkl     # Default pre-trained model
│   ├── X_features.npy / y_target.npy / meta_df.npy  # Cached training features/targets
├── results/                           # Evaluation outputs land here (auto-created if missing)
└── 2026_2027_Internal_Hackathon_1.pdf # Hackathon problem statement
```

## Troubleshooting

| Issue | Likely cause / fix |
|---|---|
| "Delivery datasets not found" error in the UI | You're not running `streamlit run` from the project root, or a CSV is missing from `Datasets/IPL DELIVERIES/`. |
| "Specified model file path does not exist" warning in sidebar | The Active Model Path points to a `.pkl` that doesn't exist yet — either retrain a model or select one from the Model UI dropdown. |
| "Cricsheet match files missing" error when running Evaluation | `Datasets/ipl_json/` is empty or missing — this folder is only needed for the Evaluate feature, not for basic recommendations. |
| App is slow to start the first time | The first load parses all delivery CSVs and JSON match files and caches them (`@st.cache_data` / `@st.cache_resource`); subsequent interactions are much faster. |
| `pip install` fails on `jax`/`numpyro`/`pymc` | These are only used for exploratory modeling in the notebook, not by `ui_app.py`. You can remove them from `requirements.txt` if you just want to run the Streamlit app. |

## Notes for contributors

- The core scoring logic (how raw ball-by-ball data is converted into Dream11-style
  fantasy points) lives in `match_dreamscore()` near the top of `ui_app.py`.
- Feature engineering for the regression model (venue form, recent form, head-to-head,
  etc.) is in `create_input_fast()` and related helper functions in the same file.
- `player_role_generator.py` is a one-off script used to (re)generate
  `csv files/player_behaviour_data.csv` and `player_roles.csv` from the raw
  deliveries data — you generally won't need to re-run it unless new player names
  appear that aren't yet classified.
