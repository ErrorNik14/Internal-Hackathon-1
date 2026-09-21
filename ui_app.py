"""Fantasy XI recommender, evaluator, and model trainer.
Run with: streamlit run ui_application_stat.py
"""

import json
import math
import os
import pickle
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
import streamlit as st
from thefuzz import process

# Config
st.set_page_config(page_title="Fantasy XI Predictor & Recommender", layout="wide")

DATASET_DIR = Path("./Datasets/IPL DELIVERIES/")
CRICSHEET_DIR = Path("./Datasets/ipl_json/")
SEASONS = [2022, 2023, 2024, 2025, 2026]
BEHAVIOUR_PATH = Path("./csv files/player_behaviour_data.csv")
ROLES_PATH = Path("./csv files/player_roles.csv")
MODEL_PATH = Path("./Stat_Features/Dream11_Team_Regressor.pkl")
MODELS_DIR = Path("./Stat_Features")
RESULTS_DIR = Path("./results")

HOME_GROUND = {
    'MI': 'Wankhede Stadium, Mumbai',
    'PBKS': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur, Chandigarh',
    'GT': 'Narendra Modi Stadium, Ahmedabad',
    'RCB': 'M.Chinnaswamy Stadium, Bengaluru',
    'RR': 'Sawai Mansingh Stadium, Jaipur',
    'SRH': 'Rajiv Gandhi International Stadium, Hyderabad',
    'DC': 'Arun Jaitley Stadium, Delhi',
    'LSG': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'KKR': 'Eden Gardens, Kolkata',
    'CSK': 'MA Chidambaram Stadium, Chennai'
}

XI = 11
POOL_WINDOW = 5
POOL_MIN = 15


################################################################################
#                              Scoring Function                                #
################################################################################
def match_dreamscore(df: pd.DataFrame):
    scores = defaultdict(int)
    runs = defaultdict(int)
    balls_faced = defaultdict(int)
    conceded = defaultdict(int)
    legal_balls = defaultdict(int)
    wickets = defaultdict(int)
    catches = defaultdict(int)
    out = set()

    for row in df.itertuples(index=False):
        # Striker score
        runs[row.striker] += row.runs_of_bat
        scores[row.striker] += row.runs_of_bat
        if row.runs_of_bat == 4:
            scores[row.striker] += 1
        elif row.runs_of_bat == 6:
            scores[row.striker] += 2
        if not row.wide:
            balls_faced[row.striker] += 1

        # Bowler score
        conceded[row.bowler] += row.runs_of_bat + row.wide + row.noballs
        if not row.wide and not row.noballs:
            legal_balls[row.bowler] += 1

        # Wickets
        if isinstance(row.wicket_type, str):
            wt = row.wicket_type.lower()
            if wt in {'caught', 'bowled', 'lbw', 'stumped', 'hit wicket'}:
                wickets[row.bowler] += 1
                scores[row.bowler] += 25
                if wt in {'bowled', 'lbw'}:
                    scores[row.bowler] += 8
            if wt == 'caught' and isinstance(row.fielder, str):
                for f in row.fielder.split('/'):
                    catches[f.strip()] += 1
                    scores[f.strip()] += 8
            if isinstance(row.player_dismissed, str):
                out.add(row.player_dismissed)

    # Milestones, ducks, strike rate
    for p, r in runs.items():
        if r >= 100:
            scores[p] += 16
        elif r >= 50:
            scores[p] += 8
        elif r >= 30:
            scores[p] += 4
        if r == 0 and p in out and balls_faced[p] > 0:
            scores[p] -= 2

        bf = balls_faced[p]
        if bf >= 10:
            sr = 100 * r / bf
            if sr > 170:
                scores[p] += 6
            elif sr > 150:
                scores[p] += 4
            elif sr >= 130:
                scores[p] += 2
            elif sr < 50:
                scores[p] -= 6
            elif sr < 60:
                scores[p] -= 4
            elif sr <= 70:
                scores[p] -= 2

    # Wicket hauls and economy
    for p, lb in legal_balls.items():
        w = wickets[p]
        if w >= 5:
            scores[p] += 16
        elif w == 4:
            scores[p] += 8
        elif w == 3:
            scores[p] += 4

        if lb >= 12:
            econ = conceded[p] / (lb / 6)
            if econ < 5:
                scores[p] += 6
            elif econ < 6:
                scores[p] += 4
            elif econ <= 7:
                scores[p] += 2
            elif econ > 12:
                scores[p] -= 6
            elif econ > 11:
                scores[p] -= 4
            elif econ >= 10:
                scores[p] -= 2

    # Catch bonus
    for p, c in catches.items():
        if c >= 3:
            scores[p] += 4

    return dict(scores)


################################################################################
#                              Data Loaders                                    #
################################################################################
@st.cache_data
def load_deliveries():
    frames = []
    for y in SEASONS:
        path = DATASET_DIR / f"ipl_{y}_deliveries.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'], format='mixed')

    X_train = data[data['date'].dt.year <= 2025]
    matches = find_similar_names(X_train)

    for k in ['Mandeep', 'Hazlewood', 'Roy', 'Mitchell', 'Mitchell Marsh']:
        matches.pop(k, None)

    matches['Mayank'] = ['Mayank Agarawal']
    matches['Rahul'] = ['Rahul Tewatia']

    for key, value in matches.items():
        for col in ['striker', 'bowler', 'fielder', 'player_dismissed']:
            data.loc[data[col].isin(value), col] = key

    return data


@st.cache_resource
def load_roles(bat_min=75, bowl_max=25):
    if BEHAVIOUR_PATH.exists():
        b = pd.read_csv(BEHAVIOUR_PATH, index_col=0)
        b.columns = ["name", "pct_faced", "pct_bowled", "is_wk"]

        def derive_role(r):
            if r.is_wk:
                return "WK"
            if r.pct_faced >= bat_min:
                return "BAT"
            if r.pct_faced <= bowl_max:
                return "BOWL"
            return "AR"

        b["role"] = b.apply(derive_role, axis=1)
        return dict(zip(b["name"], b["role"]))
    elif ROLES_PATH.exists():
        r = pd.read_csv(ROLES_PATH)
        return dict(zip(r['name'], r['role']))
    return {}


@st.cache_resource
def load_cricsheet_data():
    if not CRICSHEET_DIR.exists():
        return pd.DataFrame()

    match_list = []
    for file_name in os.listdir(CRICSHEET_DIR):
        if file_name.endswith('.json'):
            file_path = CRICSHEET_DIR / file_name
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    info = data.get('info', {})
                    teams = info.get('teams', [])
                    dates = info.get('dates', ['Unknown'])
                    players_dict = info.get('players', {})

                    team_1_name = teams[0] if len(teams) > 0 else "Unknown Team 1"
                    team_2_name = teams[1] if len(teams) > 1 else "Unknown Team 2"

                    match_list.append({
                        "match_id": file_name.replace('.json', ''),
                        "date": dates[0],
                        "team_1": team_1_name,
                        "team_2": team_2_name,
                        "team_1_players": players_dict.get(team_1_name, []),
                        "team_2_players": players_dict.get(team_2_name, [])
                    })
                except Exception:
                    continue

    df_c = pd.DataFrame(match_list)
    if df_c.empty:
        return df_c
    df_c['date'] = pd.to_datetime(df_c['date'])
    df_c = df_c.sort_values('date').reset_index(drop=True)
    return df_c[df_c['date'] >= '2022-01-01']


def align_cricsheet_names(df_cricsheet, df_original, threshold=90):
    reference_names = set()
    for col in ['striker', 'bowler', 'fielder', 'player_dismissed']:
        if col in df_original.columns:
            reference_names.update(df_original[col].dropna().unique())
    reference_names.discard('m')
    reference_names_list = list(reference_names)

    cricsheet_names = set()
    for player_list in df_cricsheet['team_1_players'].dropna():
        if isinstance(player_list, list):
            cricsheet_names.update(player_list)
    for player_list in df_cricsheet['team_2_players'].dropna():
        if isinstance(player_list, list):
            cricsheet_names.update(player_list)

    name_mapping = {}
    for name in cricsheet_names:
        best_match, score = process.extractOne(name, reference_names_list)
        if score >= threshold:
            name_mapping[name] = best_match
        else:
            name_mapping[name] = name

    df_mapped = df_cricsheet.copy()

    def replace_names(players):
        if isinstance(players, list):
            return [name_mapping.get(p, p) for p in players]
        return players

    df_mapped['team_1_players'] = df_mapped['team_1_players'].apply(replace_names)
    df_mapped['team_2_players'] = df_mapped['team_2_players'].apply(replace_names)

    name_mapping['K Yadav'] = 'Kuldeep Yadav'
    name_mapping['RA Jadeja'] = 'Ravindra Jadeja'

    return df_mapped, name_mapping


def update_roles_dictionary(roles_dict, name_mapping_dict, threshold=90):
    target_names = list(set(name_mapping_dict.values()))
    updated_roles = {}
    for current_name, role in roles_dict.items():
        best_match, score = process.extractOne(current_name, target_names)
        if score >= threshold:
            updated_roles[best_match] = role
        else:
            updated_roles[current_name] = role
    return updated_roles


@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        print("Loading:", path)
        return pickle.load(f)


################################################################################
#                          Historical Cache & Features                         #
################################################################################
@st.cache_data
def build_historical_cache(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    match_scores_lookup = {}
    for match_date, m_df in df.groupby('date'):
        match_scores_lookup[match_date] = match_dreamscore(m_df)

    venue_dates_lookup = df.groupby('venue')['date'].unique().to_dict()

    team_df = df[['season', 'date', 'batting_team', 'bowling_team']].drop_duplicates()
    teams_long = pd.concat([
        team_df[['season', 'date', 'batting_team']].rename(columns={'batting_team': 'team'}),
        team_df[['season', 'date', 'bowling_team']].rename(columns={'bowling_team': 'team'})
    ]).drop_duplicates().sort_values(['team', 'season', 'date']).reset_index(drop=True)

    teams_long['match_num'] = teams_long.groupby(['team', 'season']).cumcount() + 1

    return match_scores_lookup, venue_dates_lookup, teams_long


def compute_team_vs_team_feature(player_team, opposing_team, past_df, match_scores_lookup, c=20.0):
    h2h_dates = past_df[
        ((past_df['batting_team'] == player_team) & (past_df['bowling_team'] == opposing_team)) |
        ((past_df['batting_team'] == opposing_team) & (past_df['bowling_team'] == player_team))
    ]['date'].unique()

    team_pts, opp_pts = 0, 0
    for d in h2h_dates:
        m_scores = match_scores_lookup.get(d, {})
        m_df = past_df[past_df['date'] == d]

        pt_roster = set(m_df[m_df['batting_team'] == player_team]['striker']).union(
            set(m_df[m_df['bowling_team'] == player_team]['bowler']))
        ot_roster = set(m_df[m_df['batting_team'] == opposing_team]['striker']).union(
            set(m_df[m_df['bowling_team'] == opposing_team]['bowler']))

        for p, score in m_scores.items():
            if p in pt_roster:
                team_pts += score
            elif p in ot_roster:
                opp_pts += score

    return math.log((team_pts + (25 * c)) / (opp_pts + (25 * c)))


def create_input_fast(player_name, opposing_players, player_team, opposing_team,
                      current_date, past_df, home_ground, match_scores_lookup,
                      venue_dates_lookup, teams_long, f4_precomputed, c=20.0):
    current_date = pd.to_datetime(current_date)

    # Feature 1: Average venue points
    def get_avg_venue_pts(venue):
        if not venue or venue not in venue_dates_lookup:
            return 0.0
        past_v_dates = [d for d in venue_dates_lookup[venue] if d < current_date]
        if not past_v_dates:
            return 0.0
        return sum(match_scores_lookup[d].get(player_name, 0) for d in past_v_dates) / len(past_v_dates)

    avg_home = get_avg_venue_pts(home_ground.get(player_team))
    avg_away = get_avg_venue_pts(home_ground.get(opposing_team))
    f1 = (avg_home + avg_away) / 2.0

    # Feature 2: Match Serial Moving Average
    team_schedule = teams_long[teams_long['team'] == player_team]
    curr_match_row = team_schedule[team_schedule['date'] == current_date]

    if not curr_match_row.empty:
        curr_season = curr_match_row.iloc[0]['season']
        N = curr_match_row.iloc[0]['match_num']
    else:
        past_team = team_schedule[team_schedule['date'] < current_date]
        curr_season = past_team['season'].max() if not past_team.empty else current_date.year
        N = len(past_team[past_team['season'] == curr_season]) + 1

    target_points = []
    for season, group in team_schedule.groupby('season'):
        if season == curr_season:
            mask = (group['match_num'] >= N - 3) & (group['match_num'] <= N - 1)
        else:
            mask = (group['match_num'] >= N - 1) & (group['match_num'] <= N + 1)

        for d in group[mask]['date']:
            if d < current_date:
                target_points.append(match_scores_lookup[d].get(player_name, 0))

    f2 = sum(target_points) / len(target_points) if target_points else 0.0

    # Feature 3: Player vs Opposing Players
    p_mask = (past_df['striker'] == player_name) | \
             (past_df['bowler'] == player_name) | \
             (past_df['player_dismissed'] == player_name) | \
             (past_df['fielder'].str.contains(player_name, na=False, regex=False))

    p_past_df = past_df[p_mask]
    player_vs_opp_pts, opp_vs_player_pts = 0, 0

    if not p_past_df.empty:
        for opp in opposing_players:
            opp_mask = ((p_past_df['striker'] == player_name) & (p_past_df['bowler'] == opp)) | \
                       ((p_past_df['striker'] == opp) & (p_past_df['bowler'] == player_name)) | \
                       ((p_past_df['player_dismissed'] == player_name) & (p_past_df['fielder'].str.contains(opp, na=False, regex=False))) | \
                       ((p_past_df['player_dismissed'] == opp) & (p_past_df['fielder'].str.contains(player_name, na=False, regex=False)))

            subset = p_past_df[opp_mask]
            if not subset.empty:
                pts = match_dreamscore(subset)
                player_vs_opp_pts += pts.get(player_name, 0)
                opp_vs_player_pts += pts.get(opp, 0)

    f3 = math.log((player_vs_opp_pts + c) / (opp_vs_player_pts + c))
    f4 = f4_precomputed

    return [f1, f2, f3, f4]


################################################################################
#                           Optimization Solver                                #
################################################################################
def select_optimal_xi(players_data):
    N = len(players_data)
    if N < XI:
        return [p['name'] for p in players_data]

    c = [-p['score'] for p in players_data]
    integrality = np.ones(N)
    bounds = [(0, 1) for _ in range(N)]

    A_ub, b_ub = [], []

    for r in ['BAT', 'BOWL', 'WK', 'AR']:
        r_indices = [i for i, p in enumerate(players_data) if p['role'] == r]
        if r_indices:
            row_max = [0] * N
            for idx in r_indices:
                row_max[idx] = 1
            A_ub.append(row_max)
            b_ub.append(8)

            row_min = [0] * N
            for idx in r_indices:
                row_min[idx] = -1
            A_ub.append(row_min)
            b_ub.append(-1)

    teams = list(set(p['team'] for p in players_data))
    for t in teams:
        t_indices = [i for i, p in enumerate(players_data) if p['team'] == t]
        if t_indices:
            row_team = [0] * N
            for idx in t_indices:
                row_team[idx] = -1
            A_ub.append(row_team)
            b_ub.append(-1)

    A_eq = [[1] * N]
    b_eq = [XI]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, integrality=integrality, method='highs'
    )

    if res.success:
        selected_indices = np.where(np.round(res.x) == 1)[0]
        return [players_data[i]['name'] for i in selected_indices]
    else:
        sorted_p = sorted(players_data, key=lambda x: x['score'], reverse=True)
        return [p['name'] for p in sorted_p[:XI]]


################################################################################
#                         Recommender & Evaluator                              #
################################################################################
def find_similar_names(df, threshold=90):
    name_columns = ['striker', 'bowler', 'fielder', 'player_dismissed']
    all_names = pd.Series(dtype=str)
    for col in name_columns:
        if col in df.columns:
            all_names = pd.concat([all_names, df[col].dropna()])
    unique_names = all_names.unique().tolist()

    matches = {}
    seen_pairs = set()
    for name in unique_names:
        if '/' in name or '(sub)' in name or name == 'm':
            continue
        close_matches = []
        for match_name, score in process.extract(name, unique_names, limit=5):
            if score >= threshold and match_name != name and '/' not in match_name:
                pair = frozenset([name, match_name])
                if pair not in seen_pairs and match_name != 'm':
                    close_matches.append(match_name)
                    seen_pairs.add(pair)
        if close_matches:
            matches[name] = close_matches
    return matches


def get_squad_pool(df, team, match_date):
    match_date = pd.to_datetime(match_date)
    past = df[(df['date'] < match_date) & ((df['batting_team'] == team) | (df['bowling_team'] == team))]
    match_dates = past['date'].unique()
    if len(match_dates) == 0:
        return []

    recent_dates = match_dates[-POOL_WINDOW:]
    recent_df = past[past['date'].isin(recent_dates)]
    strikers = set(recent_df[recent_df['batting_team'] == team]['striker'])
    bowlers = set(recent_df[recent_df['bowling_team'] == team]['bowler'])
    pool = list(strikers | bowlers)

    window = POOL_WINDOW
    while len(pool) < POOL_MIN and window < len(match_dates):
        window += 2
        recent_dates = match_dates[-window:]
        recent_df = past[past['date'].isin(recent_dates)]
        strikers = set(recent_df[recent_df['batting_team'] == team]['striker'])
        bowlers = set(recent_df[recent_df['bowling_team'] == team]['bowler'])
        pool = list(strikers | bowlers)

    return pool


def recommend_xi(team_1, team_2, match_date, model, df, roles):
    match_date = pd.to_datetime(match_date)
    past_df = df[df['date'] < match_date]

    squad1 = get_squad_pool(df, team_1, match_date)
    squad2 = get_squad_pool(df, team_2, match_date)

    if not squad1 or not squad2:
        raise ValueError(f"Insufficient historical match data before {match_date.strftime('%Y-%m-%d')}.")

    match_scores_lookup, venue_dates_lookup, teams_long = build_historical_cache(df)

    f4_t1 = compute_team_vs_team_feature(team_1, team_2, past_df, match_scores_lookup)
    f4_t2 = -f4_t1

    player_pool = []
    feature_store = {}

    for player in squad1:
        feats = create_input_fast(
            player, squad2, team_1, team_2, match_date, past_df, HOME_GROUND,
            match_scores_lookup, venue_dates_lookup, teams_long, f4_precomputed=f4_t1
        )
        pred_score = float(model.predict([feats])[0])
        role = roles.get(player, 'AR')
        player_pool.append({'name': player, 'score': pred_score, 'role': role, 'team': team_1})
        feature_store[player] = feats

    for player in squad2:
        feats = create_input_fast(
            player, squad1, team_2, team_1, match_date, past_df, HOME_GROUND,
            match_scores_lookup, venue_dates_lookup, teams_long, f4_precomputed=f4_t2
        )
        pred_score = float(model.predict([feats])[0])
        role = roles.get(player, 'AR')
        player_pool.append({'name': player, 'score': pred_score, 'role': role, 'team': team_2})
        feature_store[player] = feats

    selected_names = select_optimal_xi(player_pool)
    pool_dict = {p['name']: p for p in player_pool}

    rows = []
    for name in selected_names:
        p_data = pool_dict[name]
        f = feature_store[name]
        justification = f"Player {name} has scored an average of {f[0]:.1f} points, in his hometown venue of {HOME_GROUND[pool_dict[name]['team']]}.\n"
        justification += f"For that season, he has on average {f[1]:.1f} points, in moving windows.\n"
        if f[2]>0:
            justification += f"He seems to have a positive coefficient ({f[2]:.2f}) against his opponents, scoring more than he has let them score.  \n"
        else:
            justification += f"He seems to have a negative coefficient ({f[2]:.2f}) against his opponents, opponents scoring more than he himself does.  \n"

        if f[3]>0:
            justification += f"His team {pool_dict[name]['team']} reflects a positive coefficient ({f[3]:.2f}) against other teams, opposing teams are bested more than by this team.  \n"
        else:
            justification += f"His team {pool_dict[name]['team']} reflects a negative coefficient ({f[3]:.2f}) against other teams, and thus, is bested by opposing teams more often than not.  \n"
        
        # justification = f"Venue Avg Pts: {f[0]:.1f} | Season Moving Avg: {f[1]:.1f} | Player H2H Ratio: {f[2]:.2f} | Team H2H Ratio: {f[3]:.2f}"
        rows.append({
            'name': name,
            'role': p_data['role'],
            'team': p_data['team'],
            'pred_points': round(p_data['score'], 1),
            'justification': justification
        })

    return pd.DataFrame(rows).sort_values('pred_points', ascending=False).reset_index(drop=True)


def evaluate_2026_season(df_deliveries, df_cricsheet_mapped, model, roles, c=20.0):
    df_deliveries = df_deliveries.copy()
    df_deliveries['date'] = pd.to_datetime(df_deliveries['date'])

    df_cricsheet_mapped = df_cricsheet_mapped.copy()
    df_cricsheet_mapped['date'] = pd.to_datetime(df_cricsheet_mapped['date'])

    df_2026_matches = df_cricsheet_mapped[df_cricsheet_mapped['date'].dt.year == 2026].sort_values('date').reset_index(drop=True)

    match_scores_lookup, venue_dates_lookup, teams_long = build_historical_cache(df_deliveries)

    match_overlaps = []
    match_maes = []
    results = []

    for row in df_2026_matches.itertuples(index=False):
        match_date = row.date
        t1, t2 = row.team_1, row.team_2
        squad1, squad2 = row.team_1_players, row.team_2_players

        past_df = df_deliveries[df_deliveries['date'] < match_date]

        f4_t1 = compute_team_vs_team_feature(t1, t2, past_df, match_scores_lookup, c=c)
        f4_t2 = -f4_t1

        pred_player_pool = []

        for player in squad1:
            feats = create_input_fast(
                player, squad2, t1, t2, match_date, past_df, HOME_GROUND,
                match_scores_lookup, venue_dates_lookup, teams_long, f4_precomputed=f4_t1, c=c
            )
            pred_score = float(model.predict([feats])[0])
            pred_player_pool.append({
                'name': player, 'score': pred_score,
                'role': roles.get(player, 'BAT'), 'team': t1
            })

        for player in squad2:
            feats = create_input_fast(
                player, squad1, t2, t1, match_date, past_df, HOME_GROUND,
                match_scores_lookup, venue_dates_lookup, teams_long, f4_precomputed=f4_t2, c=c
            )
            pred_score = float(model.predict([feats])[0])
            pred_player_pool.append({
                'name': player, 'score': pred_score,
                'role': roles.get(player, 'BAT'), 'team': t2
            })

        predicted_xi = select_optimal_xi(pred_player_pool)
        actual_scores = match_scores_lookup.get(match_date, {})

        actual_player_pool = [
            {'name': p['name'], 'score': actual_scores.get(p['name'], 0), 'role': p['role'], 'team': p['team']}
            for p in pred_player_pool
        ]
        actual_xi = select_optimal_xi(actual_player_pool)

        overlap_count = len(set(predicted_xi) & set(actual_xi))
        actual_pts_pred_xi = sum(actual_scores.get(p, 0) for p in predicted_xi)
        actual_pts_act_xi = sum(actual_scores.get(p, 0) for p in actual_xi)
        mae = abs(actual_pts_pred_xi - actual_pts_act_xi)

        match_overlaps.append(overlap_count)
        match_maes.append(mae)

        results.append({
            'date': match_date.strftime('%Y-%m-%d'),
            'teams': f"{t1} vs {t2}",
            'overlap_out_of_11': overlap_count,
            'pred_xi_pts': round(actual_pts_pred_xi, 1),
            'actual_xi_pts': round(actual_pts_act_xi, 1),
            'mae': round(mae, 2)
        })

    avg_overlap = np.mean(match_overlaps) if match_overlaps else 0.0
    avg_mae = np.mean(match_maes) if match_maes else 0.0

    return pd.DataFrame(results), avg_overlap, avg_mae


################################################################################
#                         Model Retraining Pipeline                            #
################################################################################
def compute_match_p2p_scores(m_df):
    """Calculates direct player-vs-player Dream11 interaction points for a single match."""
    p2p = defaultdict(float)
    for row in m_df.itertuples(index=False):
        striker = row.striker
        bowler = row.bowler
        
        r = row.runs_of_bat
        pts_bat = r + (1 if r == 4 else (2 if r == 6 else 0))
        p2p[(striker, bowler)] += pts_bat
        
        if isinstance(row.wicket_type, str):
            wt = row.wicket_type.lower()
            if wt in {'caught', 'bowled', 'lbw', 'stumped', 'hit wicket'}:
                pts_bowl = 25 + (8 if wt in {'bowled', 'lbw'} else 0)
                p2p[(bowler, striker)] += pts_bowl
            
            if wt == 'caught' and isinstance(row.fielder, str):
                dismissed = row.player_dismissed if isinstance(row.player_dismissed, str) else striker
                for f in row.fielder.split('/'):
                    p2p[(f.strip(), dismissed)] += 8
    return p2p


def _compute_f2_fast(player_name, player_team, current_date, teams_long, match_scores_lookup):
    """Computes season moving average (Feature 2) using precomputed schedule indices."""
    team_schedule = teams_long[teams_long['team'] == player_team]
    curr_match_row = team_schedule[team_schedule['date'] == current_date]

    if not curr_match_row.empty:
        curr_season = curr_match_row.iloc[0]['season']
        N = curr_match_row.iloc[0]['match_num']
    else:
        past_team = team_schedule[team_schedule['date'] < current_date]
        curr_season = past_team['season'].max() if not past_team.empty else current_date.year
        N = len(past_team[past_team['season'] == curr_season]) + 1

    target_points = []
    for season, group in team_schedule.groupby('season'):
        if season == curr_season:
            mask = (group['match_num'] >= N - 3) & (group['match_num'] <= N - 1)
        else:
            mask = (group['match_num'] >= N - 1) & (group['match_num'] <= N + 1)

        for d in group[mask]['date']:
            if d < current_date:
                target_points.append(match_scores_lookup.get(d, {}).get(player_name, 0))

    return sum(target_points) / len(target_points) if target_points else 0.0


def _update_accumulators(m_df, m_scores, venue_player_pts, p2p_cum_pts, team_h2h_cum_pts):
    """Updates running stats after feature extraction for the current match date."""
    if 'venue' in m_df.columns and not m_df['venue'].empty:
        venue = m_df['venue'].iloc[0]
        if pd.notna(venue):
            for player, score in m_scores.items():
                venue_player_pts[venue][player].append(score)
                
    teams = list(set(m_df['batting_team'].dropna().tolist() + m_df['bowling_team'].dropna().tolist()))
    if len(teams) >= 2:
        t1, t2 = teams[0], teams[1]
        h2h_key = tuple(sorted([t1, t2]))
        
        t1_roster = set(m_df[m_df['batting_team'] == t1]['striker']).union(
            set(m_df[m_df['bowling_team'] == t1]['bowler']))
        t2_roster = set(m_df[m_df['batting_team'] == t2]['striker']).union(
            set(m_df[m_df['bowling_team'] == t2]['bowler']))
            
        for p, score in m_scores.items():
            if p in t1_roster:
                team_h2h_cum_pts[h2h_key][t1] += score
            elif p in t2_roster:
                team_h2h_cum_pts[h2h_key][t2] += score

    match_p2p = compute_match_p2p_scores(m_df)
    for pair, pts in match_p2p.items():
        p2p_cum_pts[pair] += pts


def retrain_model_pipeline(df_deliveries, train_start=None, train_end=None, model_label="Dream11_Team_Regressor"):
    """Single-pass fast model retraining pipeline using chronological accumulators with date windowing."""
    df = df_deliveries.copy()
    df['date'] = pd.to_datetime(df['date'])

    train_start_dt = pd.Timestamp(train_start) if train_start else None
    train_end_dt = pd.Timestamp(train_end) if train_end else None
    
    # Precompute match scores and schedule lookups across full history
    match_scores_lookup, venue_dates_lookup, teams_long = build_historical_cache(df)
    
    unique_dates = sorted(df['date'].unique())
    matches_by_date = {d: m_df for d, m_df in df.groupby('date')}
    
    venue_player_pts = defaultdict(lambda: defaultdict(list))
    p2p_cum_pts = defaultdict(float)
    team_h2h_cum_pts = defaultdict(lambda: defaultdict(float))
    
    X_list, y_list = [], []
    c_val = 20.0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_dates = len(unique_dates)
    
    for idx, m_date in enumerate(unique_dates):
        if idx % 10 == 0 or idx == total_dates - 1:
            progress_bar.progress(min(1.0, (idx + 1) / total_dates))
            status_text.text(f"Extracting historical features from match date {idx+1}/{total_dates}...")
            
        m_df = matches_by_date[m_date]
        m_scores = match_scores_lookup.get(m_date, {})
        players = list(m_scores.keys())
        if not players:
            continue
            
        teams = list(set(m_df['batting_team'].dropna().tolist() + m_df['bowling_team'].dropna().tolist()))
        if len(teams) < 2:
            _update_accumulators(m_df, m_scores, venue_player_pts, p2p_cum_pts, team_h2h_cum_pts)
            continue
            
        # Check if match date is within selected training range
        in_range = True
        if train_start_dt and m_date < train_start_dt:
            in_range = False
        if train_end_dt and m_date > train_end_dt:
            in_range = False

        if in_range:
            t1, t2 = teams[0], teams[1]
            h2h_key = tuple(sorted([t1, t2]))
            
            # Feature 4 (Team H2H)
            t1_pts = team_h2h_cum_pts[h2h_key][t1]
            t2_pts = team_h2h_cum_pts[h2h_key][t2]
            f4_t1 = math.log((t1_pts + (25 * c_val)) / (t2_pts + (25 * c_val)))
            
            for p in players:
                p_bat = m_df[m_df['striker'] == p]['batting_team']
                p_team = p_bat.iloc[0] if not p_bat.empty else t1
                opp_team = t2 if p_team == t1 else t1
                opp_players = [op for op in players if op != p]
                f4 = f4_t1 if p_team == t1 else -f4_t1
                
                # Feature 1 (Venue Avg)
                v_home = HOME_GROUND.get(p_team)
                v_away = HOME_GROUND.get(opp_team)
                home_list = venue_player_pts[v_home][p] if v_home else []
                away_list = venue_player_pts[v_away][p] if v_away else []
                
                avg_home = sum(home_list) / len(home_list) if home_list else 0.0
                avg_away = sum(away_list) / len(away_list) if away_list else 0.0
                f1 = (avg_home + avg_away) / 2.0
                
                # Feature 2 (Season Moving Avg)
                f2 = _compute_f2_fast(p, p_team, m_date, teams_long, match_scores_lookup)
                
                # Feature 3 (Player H2H)
                p_vs_opp = sum(p2p_cum_pts[(p, opp)] for opp in opp_players)
                opp_vs_p = sum(p2p_cum_pts[(opp, p)] for opp in opp_players)
                f3 = math.log((p_vs_opp + c_val) / (opp_vs_p + c_val))
                
                X_list.append([f1, f2, f3, f4])
                y_list.append(m_scores[p])
            
        # Update running state for all dates chronologically
        _update_accumulators(m_df, m_scores, venue_player_pts, p2p_cum_pts, team_h2h_cum_pts)

    if not X_list:
        st.error(f"No match samples found between {train_start} and {train_end}.")
        return None, None

    status_text.text(f"Fitting Linear Regression model on {len(X_list)} samples...")
    model = LinearRegression()
    model.fit(X_list, y_list)

    clean_label = model_label.strip().replace(" ", "_")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"{clean_label}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    progress_bar.progress(1.0)
    status_text.text(f"Model trained successfully! Saved to {save_path}")
    return save_path, model


################################################################################
#                              Streamlit App UI                                #
################################################################################
def current_model():
    path = Path(st.session_state.get('active_model_path', str(MODEL_PATH)))
    if not path.exists():
        return None
    return load_model(str(path))


with st.sidebar:
    st.header("Configuration")
    
    if 'active_model_path' not in st.session_state:
        st.session_state['active_model_path'] = str(MODEL_PATH)

    def sync_manual_input():
        st.session_state['active_model_path'] = st.session_state['model_path_widget']

    st.text_input(
        "Active Model Path",
        value=st.session_state['active_model_path'],
        key="model_path_widget",
        on_change=sync_manual_input
    )

    if not Path(st.session_state["active_model_path"]).exists():
        st.warning("Specified model file path does not exist.")

st.title("Fantasy XI Predictor & Recommender")
product_tab, model_ui_tab = st.tabs(["Product UI", "Model UI"])


with product_tab:
    deliveries_df = load_deliveries()
    roles_map = load_roles()

    if deliveries_df.empty:
        st.error("Delivery datasets not found. Please verify dataset files in `./Datasets/IPL DELIVERIES/`.")
    else:
        teams = sorted(set(deliveries_df['batting_team'].dropna()) | set(deliveries_df['bowling_team'].dropna()))
        c1, c2, c3 = st.columns(3)
        team_1 = c1.selectbox("Team 1", teams, index=0)
        team_2 = c2.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0)
        match_date = c3.date_input("Match Date", value=date(2026, 4, 12))

        if st.button("Recommend XI", type="primary"):
            if team_1 == team_2:
                st.error("Please select two distinct teams.")
            else:
                model = current_model()
                if model is None:
                    st.error("Model file not loaded. Verify the model path in the sidebar.")
                    st.stop()

                t0 = time.perf_counter()
                try:
                    xi = recommend_xi(team_1, team_2, match_date, model, deliveries_df, roles_map)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                elapsed = time.perf_counter() - t0

                if elapsed > 10:
                    st.warning(f"Computation completed in {elapsed:.1f}s (exceeded 10s target).")

                roles_summary = xi['role'].value_counts().to_dict()
                split_summary = xi['team'].value_counts().to_dict()

                m1, m2, m3 = st.columns(3)
                m1.metric("Expected Total Points", f"{xi['pred_points'].sum():.0f}")
                m2.metric("Role Breakdown", ", ".join(f"{v} {k}" for k, v in roles_summary.items()))
                m3.metric("Team Split", " / ".join(f"{k}: {v}" for k, v in split_summary.items()))

                st.dataframe(
                    xi[['name', 'role', 'team']].rename(columns={
                        'name': 'Player', 'role': 'Role', 'team': 'Team',
                    }),
                    hide_index=True, use_container_width=True
                )

                st.subheader("Player Insights & Statistical Breakdown")
                for r in xi.itertuples(index=False):
                    with st.expander(f"{r.name} ({r.role} - {r.team})"):
                        st.text(r.justification)

                st.caption(f"Recommendation computed in {elapsed:.2f} seconds.")


with model_ui_tab:
    active_path = st.session_state.get('active_model_path', str(MODEL_PATH))
    st.info(f"**Currently Active Model for Evaluation & Inference:** `{active_path}`")

    st.subheader("Model Retraining")
    st.write("Train a new Linear Regression model directly from historical ball-by-ball delivery data.")

    model_label_input = st.text_input("Model Output Label", value="Dream11_Team_Regressor")

    c1, c2 = st.columns(2)
    train_start = c1.date_input("Train From Date", value=date(2022, 1, 1))
    train_end = c2.date_input("Train To Date", value=date(2025, 12, 31))

    if st.button("Start Retraining Pipeline", type="primary"):
        deliveries_df = load_deliveries()
        if deliveries_df.empty:
            st.error("Delivery datasets not found in `./Datasets/IPL DELIVERIES/`.")
            st.stop()

        with st.spinner("Executing feature extraction and model fitting..."):
            save_path, model = retrain_model_pipeline(
                df_deliveries=deliveries_df,
                train_start=train_start,
                train_end=train_end,
                model_label=model_label_input
            )
            if save_path:
                st.session_state["active_model_path"] = str(save_path)
                st.success(f"Model retrained and saved to `{save_path}`. Active model path updated!")
                st.rerun()

    st.divider()
    st.write("### Select Active Model")
    if MODELS_DIR.exists():
        model_files = [f.name for f in MODELS_DIR.glob("*.pkl")]
        if model_files:
            selected_file = st.selectbox("Select model file to set as active:", model_files)
            if st.button("Set Active Model"):
                st.session_state["active_model_path"] = str(MODELS_DIR / selected_file)
                st.success(f"Active model updated to: `{MODELS_DIR / selected_file}`")
                st.rerun()
        else:
            st.info("No saved model pickle files found in `./Stat_Features/`.")
    else:
        st.info("Directory `./Stat_Features/` does not exist yet.")

    st.subheader("Evaluate Active Model on Season 2026")
    c1, c2 = st.columns(2)
    test_start = c1.date_input("Test From Date", value=date(2026, 3, 28))
    test_end = c2.date_input("Test To Date", value=date(2026, 5, 31))

    if st.button("Run Evaluation", type="primary"):
        model = current_model()
        if model is None:
            st.error(f"Model file not loaded from `{active_path}`. Verify the path.")
            st.stop()

        deliveries_df = load_deliveries()
        cricsheet_df = load_cricsheet_data()

        if deliveries_df.empty:
            st.error("Delivery dataset is missing.")
            st.stop()
        if cricsheet_df.empty:
            st.error("Cricsheet match files missing in `./Datasets/ipl_json/`.")
            st.stop()

        with st.spinner("Processing match evaluations across selected timeframe..."):
            cricsheet_mapped, name_mapping = align_cricsheet_names(cricsheet_df, deliveries_df)
            roles_map = load_roles()
            updated_roles = update_roles_dictionary(roles_map, name_mapping)

            eval_df, avg_matches, avg_mae = evaluate_2026_season(
                df_deliveries=deliveries_df,
                df_cricsheet_mapped=cricsheet_mapped,
                model=model,
                roles=updated_roles,
                c=20.0
            )

        if not eval_df.empty and 'date' in eval_df.columns:
            d = pd.to_datetime(eval_df['date'])
            eval_df = eval_df[(d >= pd.Timestamp(test_start)) & (d <= pd.Timestamp(test_end))]
            if not eval_df.empty:
                avg_matches = eval_df['overlap_out_of_11'].mean()
                avg_mae = eval_df['mae'].mean()

        if eval_df.empty:
            st.info("No matches found within the selected date window.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Average Matched Players", f"{avg_matches:.2f} / 11")
            c2.metric("Mean Absolute Error (MAE)", f"{avg_mae:.2f}")
            c3.metric("Evaluated Matches", len(eval_df))

            st.dataframe(
                eval_df.rename(columns={
                    'date': 'Date', 'teams': 'Teams',
                    'overlap_out_of_11': 'Matched Players (/11)',
                    'pred_xi_pts': 'Predicted XI Score',
                    'actual_xi_pts': 'Actual XI Score',
                    'mae': 'MAE'
                }),
                hide_index=True, use_container_width=True
            )

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RESULTS_DIR / f"predictions_{test_start}_{test_end}.csv"
            eval_df.to_csv(out_path, index=False)
            st.success(f"Results saved to `{out_path}`.")

            st.download_button(
                "Download Evaluation CSV",
                eval_df.to_csv(index=False).encode(),
                file_name=out_path.name,
                mime="text/csv"
            )
